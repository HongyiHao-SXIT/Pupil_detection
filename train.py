import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

# -----------------------------
# GPU 检查
# -----------------------------
def check_gpu():
    print("=== PyTorch GPU Check ===")
    print("torch version:", torch.__version__)
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
    print("=========================\n")

# -----------------------------
# 加载模型
# -----------------------------
def load_model(model_path='yolo11n.pt', device='cuda:0'):
    print("=== Loading YOLO Model ===")
    model = YOLO(model_path)
    # 强制模型移到 GPU
    model.model.to(device)
    print("First model parameter device:", next(model.model.parameters()).device)
    print("==========================\n")
    return model

# -----------------------------
# 模型训练
# -----------------------------
def train_model(model, data_yaml, project_dir, name="train", epochs=200, batch=16, workers=8, device='cuda:0'):
    print("=== Start Training ===")
    model.train(
        cfg='ultralytics/cfg/default.yaml',
        data=data_yaml,
        optimizer='SGD',
        epochs=epochs,
        project=project_dir,
        name=name,
        batch=batch,
        workers=workers,
        split='val',
        device=device
    )
    print("=====================\n")

# -----------------------------
# 模型验证
# -----------------------------
def validate_model(model, data_yaml, batch=16, device='cuda:0', save=True):
    print("=== Start Validation ===")
    results = model.val(
        data=data_yaml,
        batch=batch,
        device=device,
        save=save
    )
    print("Validation Metrics:", results.metrics)  # 输出 mAP, precision, recall
    print("========================\n")
    return results

# -----------------------------
# 模型推理
# -----------------------------
def inference(model, source, project_dir, name='predict', device='cuda:0'):
    print("=== Start Inference ===")
    results = model.predict(
        source=source,
        save=True,
        device=device,
        project=project_dir,
        name=name
    )
    print("Inference done, results saved in:", os.path.join(project_dir, name))
    print("=======================\n")
    return results

# -----------------------------
# 模型导出
# -----------------------------
def export_model(model, device='cuda:0', format='onnx', simplify=True, opset=11):
    print("=== Export Model ===")
    model.export(
        device=device,
        format=format,
        simplify=simplify,
        opset=opset
    )
    print("===================\n")

# -----------------------------
# 可视化检测热力图
# -----------------------------
def show_heatmap(image_path, results):
    print("=== Show Heatmap ===")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = torch.zeros(img.shape[:2], dtype=torch.float32)
    for box in results[0].boxes.xyxy.cpu():
        x1, y1, x2, y2 = box.int()
        heatmap[y1:y2, x1:x2] += 1

    heatmap = heatmap / heatmap.max()
    plt.imshow(img_rgb)
    plt.imshow(heatmap.numpy(), cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()
    print("====================\n")

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    check_gpu()

    # 模型路径
    model_path = 'yolo11n.pt'
    model = load_model(model_path=model_path, device='cuda:0')

    # 数据集配置
    data_yaml = r"data.yaml"
    project_dir = r"YOLO11/runs"

    # 训练
    train_model(model, data_yaml=data_yaml, project_dir=project_dir,
                name="train-200epoch-v11n-bs16", epochs=200, batch=16, workers=8, device='cuda:0')

    # 验证
    val_results = validate_model(model, data_yaml=data_yaml, batch=16, device='cuda:0', save=True)

    # 推理
    test_img = r"YOLO11/ultralytics/assets/test.jpg"
    pred_results = inference(model, source=test_img, project_dir=project_dir, name='predict', device='cuda:0')

    # 热力图可视化
    show_heatmap(test_img, pred_results)

    # 导出模型 ONNX
    export_model(model, device='cuda:0', format='onnx', simplify=True, opset=11)
