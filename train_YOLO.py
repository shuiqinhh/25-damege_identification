import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm 
import numpy as np
from datetime import datetime


# Hyperparameter configuration
class Config:
    BATCH_SIZE = 32
    EPOCHS = 64
    LR =3e-3  
    WEIGHT_DECAY = 5e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4  #数据加载线程数
    
    
    EARLY_STOPPING_PATIENCE = 10 #连续10轮无提升则停止训练
    IMG_SIZE = 640  # YOLO标准输入尺寸
    
    # 路径
    TRAIN_DIR = r"D:\DeskTop\demage_identification\training\data_splitted\training_data"           #训练数据
    VAL_DIR = r"D:\DeskTop\demage_identification\training\data_splitted\validation_data"           #验证数据
    TEST_DIR = r"D:\DeskTop\demage_identification\training\data_splitted\test_data"                #测试数据
    BEST_MODEL_PATH = r"D:\DeskTop\demage_identification\training\best_models_yolo\best_model_yolo(LR=3e-3).pth"   #保存最佳模型
    RESULTS_DIR = r'D:\DeskTop\demage_identification\training\results_yolo'                        #结果保存文件夹
    RESULTS_FILE = r'D:\DeskTop\demage_identification\training\results_yolo\results(LR=3e-3).xlsx'  #训练结果Excel文件
    
    

#数据加载模块
class LeafDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform
        self.samples = []
        self.class_counts = {0: 0, 1: 0}
        
        if not os.path.exists(dir):
            raise ValueError(f"Directory not found: {dir}")
        
        for img_name in os.listdir(dir):         #遍历图像文件
            img_path = os.path.join(dir,img_name)
            if not os.path.isfile(img_path) or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): #过滤命名或格式不满足规则的图像
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except:
                print(f'跳过无效图像: {img_path}') 
                continue               

        #分配标签值
            if img_name.startswith('Healthy'):
                label = 0
            elif img_name.startswith('Faulty'):
                label = 1
            else:
                continue
                
            self.samples.append((os.path.join(dir, img_name), label))
            self.class_counts[label] += 1
        
        print(f"Dataset loaded from {dir}:")
        print(f"  Healthy: {self.class_counts[0]}, Faulty: {self.class_counts[1]}")
        print(f"  Total: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)    #计算样本量大小
#加载图像并应用变换和预处理
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE), label  #图像加载失败时返回空图像，为了避免程序崩溃


# 数据增强
transform_train = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),  #调整图像大小
    transforms.ToTensor(),     #转张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #标准化
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),   #随机擦除
])
#测试集仅标准化处理
transform_test = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# YOLO基础卷积块
class ConvBNSiLU(nn.Module):
    #标准卷积+批归一化+SiLU激活（YOLOv5）
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    #有两个CSP瓶颈的模块(需学习)
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBNSiLU(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C3(nn.Module):
    #含三个卷积和多个瓶颈的CSP模块
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.conv3 = ConvBNSiLU(2 * hidden_channels, out_channels, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, groups, expansion=1.0) for _ in range(n)])
    
    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))


class SPPF(nn.Module):
    #快速空间金字塔池化
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBNSiLU(hidden_channels * 4, out_channels, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))


class YOLOClassifier(nn.Module):
    #主干YOLO分类器
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super().__init__()
        
        
        self.backbone = nn.Sequential(
        
            ConvBNSiLU(3, 64, 6, 2, 2),  
            #P1/2
            ConvBNSiLU(64, 128, 3, 2),
            C3(128, 128, n=3),
            
            # P3/8
            ConvBNSiLU(128, 256, 3, 2),
            C3(256, 256, n=6),
            
            # P4/16
            ConvBNSiLU(256, 512, 3, 2),
            C3(512, 512, n=9),
            
            # P5/32
            ConvBNSiLU(512, 1024, 3, 2),
            C3(1024, 1024, n=3),
            SPPF(1024, 1024),
        )
        
        # 融合注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, 256, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 1024, 1),
            nn.Sigmoid()
        )
        
        # 最终分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        
        features = self.backbone(x)
        
        
        att = self.attention(features)
        features = features * att
        
        
        return self.classifier(features)


# 早停机制（可防止过拟合）
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

#训练模块
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item() * labels.size(0)
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
    
    return running_loss/total, correct/total

#验证模块
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * labels.size(0)
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
    
    return running_loss/total, correct/total

#主程序模块
def main():
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.BEST_MODEL_PATH), exist_ok=True)
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = True
    
    print("加载数据集...")
    train_dataset = LeafDataset(Config.TRAIN_DIR, transform=transform_train)
    val_dataset = LeafDataset(Config.VAL_DIR, transform=transform_test)
    test_dataset = LeafDataset(Config.TEST_DIR, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                            shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                          shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                           shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # 初始化模型
    print(f"Initializing YOLO Classifier on {Config.DEVICE}...")
    model = YOLOClassifier(num_classes=2).to(Config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.LR * 0.01)
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    #早停机制实现
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, verbose=True)
    
    #保存训练历史
    history = {
        'epoch': [],
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    # Training循环
    print("\nStarting training...")
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch [{epoch+1}/{Config.EPOCHS}]")
        
        #训练参数
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE, scaler)
        
        #验证参数
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        
        #更新训练历史数据
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        #最佳模型保存逻辑
        save_model = False
        if val_acc > best_val_acc:
            save_model = True
        elif abs(val_acc - best_val_acc) < 1e-6 and val_loss < best_val_loss:
            save_model = True
            
        if save_model:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, Config.BEST_MODEL_PATH)
            print(f"✓ New best model! Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        
    
        scheduler.step()
        
        #早停情况检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 时间戳变量
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    df = pd.DataFrame(history)
    df.to_excel(Config.RESULTS_FILE, index=False)
    
    print("\nEvaluating on test set...")
    checkpoint = torch.load(Config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    target_names = ['Healthy', 'Faulty']
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\nTest Set Classification Report:")
    print(report)
    
    print(f"\nTraining completed! Results saved to {Config.RESULTS_DIR}")


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()