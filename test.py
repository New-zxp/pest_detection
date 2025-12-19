from ultralytics import YOLO
import os
import pandas as pd

def evaluate_pest_model():
    # 1. 指向训练好的最佳权重文件
    # 请根据你实际生成的路径修改，通常在 runs/pest_detection/ip102_experiment/weights/best.pt
    model_path = r"runs/pest_detection/ip102_experiment/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"错误：未找到模型文件 {model_path}，请检查路径。")
        return

    # 2. 加载模型
    model = YOLO(model_path)

    # 3. 在测试集上运行验证
    # split='test' 表示强制使用 test 文件夹的数据
    print("正在测试集上评估模型性能...")
    results = model.val(split='test')

    # 4. 打印核心指标
    print("\n" + "="*30)
    print("      模型测试性能简报")
    print("="*30)
    print(f"Top-1 准确率 (Accuracy): {results.top1:.4f}")
    print(f"Top-5 准确率 (Accuracy): {results.top5:.4f}")
    print(f"推理速度 (Speed): {results.speed['inference']:.2f} ms/张")
    print("="*30)

    # 5. 结果保存路径说明
    print(f"\n详细测试报表（含混淆矩阵、PR曲线等）已保存至: {results.save_dir}")

if __name__ == "__main__":
    evaluate_pest_model()