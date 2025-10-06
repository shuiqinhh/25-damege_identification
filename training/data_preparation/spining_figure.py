import os
from PIL import Image
import glob

def get_index_from_filename(path):
    """从文件名中提取编号"""
    filename = os.path.basename(path)
    return int(filename.split('_')[1].split('.')[0])

def rotate_and_save_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 类别列表
    classes = ['Healthy', 'Faulty']

    for cls in classes:
        # 获取原始图像路径
        file_pattern = os.path.join(input_dir, f'{cls}_*.jpg')
        image_paths = glob.glob(file_pattern)

        # 按编号排序
        image_paths.sort(key=lambda x: get_index_from_filename(x))

        # 创建输出子目录
        output_subdir = os.path.join(output_dir, cls)
        os.makedirs(output_subdir, exist_ok=True)

        current_index = 1

        for img_path in image_paths:
            # 打开原始图像
            img = Image.open(img_path)

            # 旋转角度
            angles = [0, 90, 180, 270]

            for angle in angles:
                # 旋转图像
                rotated_img = img.rotate(angle, expand=True)

                # 构建新文件名
                new_filename = f"{cls}_{current_index}.jpg"
                new_path = os.path.join(output_subdir, new_filename)

                # 保存图像
                rotated_img.save(new_path)

                current_index += 1

        print(f"类别 {cls} 增强完成，共生成 {current_index - 1} 张图像")

if __name__ == "__main__":
    input_dir = r"D:\DeskTop\demage_identification\data_preparation\data_all"          # 原始图像目录
    output_dir = r"D:\DeskTop\demage_identification\data_preparation\data_reinforced"  # 增强后图像目录

    rotate_and_save_images(input_dir, output_dir)