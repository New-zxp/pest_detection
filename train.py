from ultralytics import YOLO

def train_pest_model():
    # 1. 加载预训练的 YOLO11 分类模型 (n代表nano版，速度快；也可选 s, m, l)
    # 官方目前 YOLOv8/v9/v10/v11 的分类权重通常以 .pt 结尾
    model = YOLO("yolo11n-cls.pt") 

    # 2. 开始训练
    results = model.train(
        data="pest_classification_data", # 指向我们刚才生成的文件夹
        epochs=100,                     # 训练轮次
        imgsz=224,                      # 分类通常使用 224x224
        batch=32,                       # 批大小
        device=0,                       # 使用 GPU (若无 GPU 则设为 'cpu')
        project="runs/pest_detection",  # 保存路径
        name="ip102_experiment"         # 实验名称
    )

    # 3. 验证模型
    metrics = model.val()
    print(f"Top-1 准确率: {metrics.top1}")

if __name__ == "__main__":
    train_pest_model()