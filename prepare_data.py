import os
import shutil

# 配置路径
base_path = 'ip102_v1.1'
images_dir = os.path.join(base_path, 'images')
output_dir = 'pest_classification_data'

def organize_yolo_data(split_file, split_name):
    txt_path = os.path.join(base_path, split_file)
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2: continue
        img_name, class_id = parts[0], parts[1]
        
        # 目标文件夹：output/train/0, output/train/1 ...
        target_dir = os.path.join(output_dir, split_name, class_id)
        os.makedirs(target_dir, exist_ok=True)
        
        src_path = os.path.join(images_dir, img_name)
        dst_path = os.path.join(target_dir, img_name)
        
        if os.path.exists(src_path):
            # 使用复制（copy）比较安全，如果空间不够可以用 move
            shutil.copy(src_path, dst_path)

# 执行整理
print("正在整理数据集...")
organize_yolo_data('train.txt', 'train')
organize_yolo_data('val.txt', 'val')
organize_yolo_data('test.txt', 'test')
print(f"数据整理完成，存放在: {output_dir}")