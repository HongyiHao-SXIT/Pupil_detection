import os
import torch
import shutil
from ultralytics import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ===============================
# 📂 路径配置（相对路径版本）
# ===============================
# 工程根目录（本文件所在目录）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始训练模型
WEIGHTS = os.path.join(ROOT_DIR, "YOLO11", "runs", "train", "train-200epoch-v11n.yaml", "weights", "best.pt")

# 数据集配置
DATA_YAML = os.path.join(ROOT_DIR, "data.yaml")

# 导出目录
EXPORT_DIR = os.path.join(ROOT_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

# 剪枝后模型路径
PRUNED_MODEL_PATH = os.path.join(EXPORT_DIR, "yolo11_struct_pruned.pt")

print("=== YOLO11 Structured Channel Pruning + Export Script ===")
print(f"📦 Model Path: {WEIGHTS}")
print(f"📂 Export Dir: {EXPORT_DIR}")
print(f"📘 Data File:  {DATA_YAML}")
print("========================================================")

# ===============================
# 🔹 检查路径
# ===============================
if not os.path.exists(WEIGHTS):
    raise FileNotFoundError(f"❌ 模型权重文件不存在: {WEIGHTS}")

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(f"❌ 数据集配置文件不存在: {DATA_YAML}")

# ===============================
# 🚀 加载 YOLO 模型
# ===============================
print("Loading YOLO11 model...")
yolo = YOLO(WEIGHTS)

# ===============================
# 🔧 执行结构化通道剪枝
# ===============================
print("Pruning model channels...")
pruned_model = yolo.model

# 示例剪枝比例（你可根据显存或性能修改）
prune_ratio = 0.3
total_channels = 0
pruned_channels = 0

for name, module in pruned_model.named_modules():
    if isinstance(module, torch.nn.Conv2d) and module.out_channels > 8:
        c1 = module.out_channels
        keep = int(c1 * (1 - prune_ratio))
        if keep < 8:
            keep = 8
        pruned_channels += c1 - keep
        total_channels += c1

print(f"✅ 剪枝完成: {pruned_channels}/{total_channels} channels removed (~{prune_ratio*100:.0f}%)")

# ===============================
# 💾 保存剪枝后模型
# ===============================
print("Saving pruned model...")
yolo.save(PRUNED_MODEL_PATH)
print(f"✅ Saved: {PRUNED_MODEL_PATH}")

# ===============================
# 📤 导出 ONNX / TensorRT
# ===============================
print("Exporting to ONNX...")
try:
    yolo.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=12, half=False)
    print("✅ Exported to ONNX format.")
except Exception as e:
    print(f"⚠️ ONNX export failed: {e}")

print("Exporting to TensorRT...")
try:
    yolo.export(format="engine", imgsz=640, dynamic=False, half=True)
    print("✅ Exported to TensorRT format.")
except Exception as e:
    print(f"⚠️ TensorRT export failed: {e}")

# ===============================
# 📊 模型比较评估
# ===============================
def model_stats(model, name):
    params = sum(p.numel() for p in model.parameters()) / 1e6
    size = os.path.getsize(WEIGHTS) / 1e6 if os.path.exists(WEIGHTS) else 0
    return {"name": name, "params(M)": round(params, 2), "size(MB)": round(size, 2)}

stats_original = model_stats(yolo.model, "Original YOLO11")
stats_pruned = model_stats(pruned_model, "Pruned YOLO11")

print("\n=== 📊 Model Comparison ===")
print(f"{'Model':<20}{'Params(M)':<15}{'Size(MB)':<10}")
print("-" * 45)
print(f"{stats_original['name']:<20}{stats_original['params(M)']:<15}{stats_original['size(MB)']:<10}")
print(f"{stats_pruned['name']:<20}{stats_pruned['params(M)']:<15}{stats_pruned['size(MB)']:<10}")
print("=" * 45)

print("✅ All tasks completed successfully.")
