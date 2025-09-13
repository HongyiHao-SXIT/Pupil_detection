from ultralytics import YOLO

model = YOLO("path/to/your/original/best.pt")   # 改成你的权重路径
print("Loaded YOLO object:", model)
# 运行一次验证（会使用 checkpoint 中记录的 data info，或你可以传 data="data.yaml"）
results = model.val()  # 输出 mAP 等指标，方便对比
print(results.metrics)  # 简要查看
