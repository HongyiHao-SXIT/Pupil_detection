import os
import time
import psutil
import torch
import numpy as np
from ultralytics import YOLO

# ==============================================================
# 配置参数
# ==============================================================
original_model_path = r"YOLO11/runs/train/train-200epoch-v11n.yaml/weights/best.pt"
pruned_model_path = r"yolo11_pruned.pt"
data_path = r"data.yaml"       # 数据集配置
test_images = 100              # 评估时抽取的图片数量
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================
# 工具函数
# ==============================================================

def get_file_size(path):
    return os.path.getsize(path) / (1024 * 1024)

def measure_inference(model, imgsz=640, num_samples=100):
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    times = []
    torch.cuda.empty_cache()

    # 预热几次
    for _ in range(5):
        _ = model(dummy_input)
    torch.cuda.synchronize()

    # 测速
    with torch.no_grad():
        for _ in range(num_samples):
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            times.append(time.time() - start)

    avg_time = np.mean(times)
    fps = 1 / avg_time
    return avg_time * 1000, fps  # 毫秒/帧率

def get_gpu_usage():
    if not torch.cuda.is_available():
        return {"memory": 0, "power": 0}
    import subprocess
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,power.draw", "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip().split("\n")[0]
        mem, power = map(float, result.split(", "))
        return {"memory": mem, "power": power}
    except Exception:
        return {"memory": 0, "power": 0}

def evaluate_model(model_path, data_path, imgsz=640):
    print(f"\n🔍 Evaluating model: {model_path}")
    model = YOLO(model_path)
    model.to(device)

    # 文件大小
    size_mb = get_file_size(model_path)

    # 性能评估（mAP、precision、recall）
    results = model.val(data=data_path, imgsz=imgsz, batch=16, device=device, plots=False, verbose=False)
    metrics = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
    }

    # 推理时间 & FPS
    avg_time, fps = measure_inference(model.model, imgsz=imgsz)

    # 资源占用
    gpu_stats = get_gpu_usage()
    cpu_percent = psutil.cpu_percent(interval=1)

    return {
        "path": model_path,
        "size_MB": round(size_mb, 2),
        "mAP50": round(metrics["mAP50"], 4),
        "mAP50-95": round(metrics["mAP50-95"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "infer_time_ms": round(avg_time, 2),
        "FPS": round(fps, 2),
        "GPU_mem_MB": gpu_stats["memory"],
        "GPU_power_W": gpu_stats["power"],
        "CPU_usage_%": cpu_percent
    }

# ==============================================================
# 主程序
# ==============================================================

def main():
    results = []
    for model_path in [original_model_path, pruned_model_path]:
        if not os.path.exists(model_path):
            print(f"❌ {model_path} not found, skipping.")
            continue
        res = evaluate_model(model_path, data_path)
        results.append(res)

    if len(results) == 2:
        print("\n===================== 📊 MODEL COMPARISON =====================")
        print(f"{'Metric':<15} | {'Original':<15} | {'Pruned':<15} | {'Δ Change':<10}")
        print("---------------------------------------------------------------")
        for key in results[0].keys():
            if key in ['path']:
                continue
            o = results[0][key]
            p = results[1][key]
            diff = round(p - o, 4) if isinstance(o, (int, float)) else "-"
            print(f"{key:<15} | {o:<15} | {p:<15} | {diff:<10}")
        print("===============================================================")
    else:
        print("⚠️ Not enough models evaluated.")

if __name__ == "__main__":
    main()
