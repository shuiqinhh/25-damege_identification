import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import json
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm 
import openpyxl



# 超参数设置
class Config:
   BATCH_SIZE = 32 #批次量大小
   EPOCHS = 64  #训练轮数
   LR = 0.  #学习率
   WEIGHT_DECAY = 1e-5 #权重衰减（正则化系数）
   DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #GPU加速
   NUM_WORKERS = 4 #线程数为4

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform
        self.samples = []                #创建储存数据路径和标签的列表
        for img_name in os.listdir(dir):
            if img_name.startswith('Healthy'):
                label = 0
            elif img_name.startswith('Faulty'):
                label = 1
            else:
                continue                       #设置健康叶片标签为0，损伤叶片标签为1，不符合命名规则的文件跳过
            self.samples.append((os.path.join(dir, img_name), label))   #将文件路径、名称和标签导入列表

    def __len__(self):
        return len(self.samples)      #遍历列表，输出数据集总样本数

    def __getitem__(self, idx):
        path, label = self.samples[idx]     #通过索引找到相应文件的路径与标签      
        image = Image.open(path).convert('RGB')      #将图片转为RGB格式
        if self.transform:                        
            image = self.transform(image)           #对图片应用预处理函数
        return image, label


# 数据在线预处理和加载
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),         #调整图片大小为224×224
    transforms.ToTensor(),                 #图片数据（RGB像素、通道数）转为张量--【通道数，高度，宽度】（C,H,W）
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),     #标准化处理
    transforms.RandomErasing(p=0.2, value='random')  # 在线数据增强：随机擦除（0.2概率随机遮挡图像）
])
#测试集预处理，参数与训练集相同，但不进行数据增强
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#神经网络架构：用到了ResNet50结合自注意机制（Attention）
class ResNet50WithAttention(nn.Module):
    def __init__(self, num_classes=2):             #分类数为2（有损伤与无损伤）
        super(ResNet50WithAttention, self).__init__()
        resnet = models.resnet50(pretrained=True)           #加载RESNET50模型，用预训练过的参数
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])    #去掉最后两层网络（全局平均池化层和全连接层），保留特征提取部分，输出特征图尺寸：(B,2048, 7, 7) （B，C，H，W），每个通道输入为1个H×W的特征图

        #引入自注意力
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # 全局平均池化：特征图由(B,2048,7,7)变为(B,2048,1,1)
            nn.Conv2d(2048, 2048 // 16, 1),     #通过1×1卷积核压缩通道数，输出（B，128，1，1）（加权）
            nn.ReLU(),                          #激活函数，负值输入输出0，其他输入输出自身值
            nn.Conv2d(2048 // 16, 2048, 1),     #通过1×1卷积核恢复通道数，输出（B,2048,1,1）
            nn.Sigmoid()                        #通过Sigmoid函数输出注意力权重，每个通道对应一个0~1间的值
        )
        #最后分类器模块
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            #全局平均池化：特征图由(B,2048,7,7)变为(B,2048,1,1)
            nn.Flatten(),                       #将多维张量转化为1维
            nn.Dropout(0.5),                    #随机丢弃50%的神经元，降低模型对训练数据的依赖
            nn.Linear(2048, num_classes)        #全连接层，输出分类结果（B，num_classes）
        )

    #前向传播
    def forward(self, x):
        features = self.feature_extractor(x)     #提取特征：依次输入（B，3，224，224），共输出（B，2048，7，7）
        attention = self.attention(features)     #计算注意力权重，输出（B，2048，1，1）
        features = features * attention          #特征值×权重
        return self.classifier(features)         #返回分类值
    

def main():
    train_dataset = CustomDataset(dir=r"D:\DeskTop\demage_identification\ing\data_splitted\training_data", transform=transform_train)    #导入数据集路径
    val_dataset = CustomDataset(dir=r"D:\DeskTop\demage_identification\ing\data_splitted\validation_data", transform=transform_test)
    test_dataset = CustomDataset(dir=r"D:\DeskTop\demage_identification\ing\data_splitted\test_data", transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,num_workers=Config.NUM_WORKERS, shuffle=True,pin_memory=True)  #创建数据加载器，shuffle=True表示打乱顺序
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS,shuffle=False,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS,shuffle=False,pin_memory=True)

    model = ResNet50WithAttention(num_classes=2).to(Config.DEVICE)  #初始化模型并移至GPU

# 引入损失函数和优化器（依旧是交叉熵损失和Adam优化器）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR,weight_decay=Config.WEIGHT_DECAY) #用Adam优化学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,factor=0.5,verbose=True) #学习率调整策略：验证损失不再下降时调整，5个轮次无改善调整，调整学习率为之前的一半，打印调整信息




# 训练与验证函数部分
# 进入循环训练并记录和导出
#创建空列表
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    best_acc = 0.0
    best_loss = float('inf')        #表示正无穷大浮点数
#训练阶段
    for epoch in range(Config.EPOCHS):        #循环遍历轮次
        model.train()                  #训练模式，用到dropout,batchnorm等
        train_ing_loss = 0.0
        train_total_samples = 0
        train_ing_correct = 0
#添加实时进度条
        train_bar = tqdm(train_loader,desc=f"Epoch[{epoch+1}/{Config.EPOCHS}] - Train", leave = True)
        for images,labels in train_bar:
           images = images.to(Config.DEVICE)
           labels = labels.to(Config.DEVICE)
        #训练逻辑
           optimizer.zero_grad()   #梯度清零，防止累积
           outputs = model(images)  #前向传播，用当前模型进行预测得到输出
           loss = criterion(outputs, labels) #用交叉熵函数计算损失（交叉熵函数在line94导入）
           loss.backward()  #反向传播，及损失函数求梯度
           optimizer.step() #用Adam优化器（原理需弄清）更新模型参数
        
           _,predicted = torch.max(outputs.data,1)        #取预测概率最大的类
           train_ing_correct += (predicted == labels).sum().item()  #计算预测正确的样本数
           train_total_samples += labels.size(0)           #获取总样本数
           train_ing_loss += loss.item()*labels.size(0)    #计算总损失（与批次量相乘）
           train_bar.set_postfix({
            'loss': train_ing_loss / train_total_samples,  
            'acc': train_ing_correct / train_total_samples       
            })                                            #更新实时进度条
        

        avg_train_loss = train_ing_loss/train_total_samples
        avg_train_acc = train_ing_correct /train_total_samples         #计算当前轮次的准确率和损失值
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)                       #添加当前轮次的准确率和损失值到列表中

#验证阶段（基本逻辑与计算与训练阶段一致）
        model.eval()                     #评估模式，关闭dropout,batchnorm等
        val_ing_loss = 0.0
        val_total_samples = 0
        val_ing_correct = 0
    

        val_bar = tqdm(val_loader,desc = f"Epoch[{epoch+1}/{Config.EPOCHS}] - Val", leave = True)

        with torch.no_grad():      #关闭梯度计算
            for images,labels in val_bar:
            
            #验证逻辑
               images = images.to(Config.DEVICE)
               labels = labels.to(Config.DEVICE)
               outputs = model(images)
               loss = criterion(outputs,labels)  

               _,predicted = torch.max(outputs.data,1)
               val_ing_correct += (predicted == labels).sum().item()
               val_total_samples += labels.size(0)
               val_ing_loss += loss.item()*labels.size(0)


               val_bar.set_postfix({
                  'loss': val_ing_loss / val_total_samples,
                  'acc': val_ing_correct / val_total_samples
               })
        avg_val_loss = val_ing_loss / val_total_samples 
        avg_val_acc = val_ing_correct / val_total_samples 
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)

    

#最佳模型计算法则（验证准确率更高或准确率相同但损失值更低）
        scheduler.step(avg_val_loss)
        if avg_val_acc > best_acc or (avg_val_acc == best_acc and avg_val_loss < best_loss) :
           best_acc = avg_val_acc
           best_loss = avg_val_loss
           torch.save(model.state_dict(), r"D:\DeskTop\demage_identification\ing\best_models\best_model(LR=3e-5,wd=1e-5).pth")  #变更学习率时记得改路径
           print(f"\033[92m Epoch {epoch+1}: 有新的最佳模型出现！验证准确率 = {avg_val_acc * 100:.2f}%\033[0m")
    

    

# 保存训练数据到 Excel文件
    df = pd.DataFrame({
       'Epoch':range(1,Config.EPOCHS + 1),
       'Train accuracy':train_acc,
       'Train loss': train_loss,
       'Validation accuracy': val_acc,
       'Validation loss':val_loss
        })
    os.makedirs(r"D:\DeskTop\demage_identification\ing\results", exist_ok=True)
    df.to_excel(r"D:\DeskTop\demage_identification\ing\results\results(LR=3e-5,wd=1e-5).xlsx", index=False) #学习率变更时记得改路径
# 测试集评估
    model.load_state_dict(torch.load(r"D:\DeskTop\demage_identification\ing\best_models\best_model(LR=3e-5,wd=1e-5).pth"))
    model.eval()       #加载最佳模型并切换评估模式

    all_preds = []    #存储预测结果
    all_labels = []   #存储真实标签

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())


# 打印分类报告（含精确率，召回率，F1分数等
    target_names = ['Healthy', 'Faulty']
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print(" 测试集分类报告：")
    print(report)

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    main()

