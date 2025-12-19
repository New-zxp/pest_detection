from ultralytics import YOLO
import PIL.Image as Image

# 1. 加载训练好的模型
model = YOLO("runs/pest_detection/ip102_experiment/weights/best.pt")

# 2. 进行预测
results = model.predict(r"C:\Users\34733\Desktop\课程\计算机视觉与图像处理\Pest-Detection\pest_classification_data\test\0\00007.jpg")

# 3. 打印结果
for result in results:
    probs = result.probs  # 获取概率对象
    print(f"最可能的类别 ID: {probs.top1}")
    print(f"置信度: {probs.top1conf.item():.4f}")