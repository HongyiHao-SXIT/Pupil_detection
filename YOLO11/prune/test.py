from ultralytics import YOLO
yolo = YOLO("./yolov8n.pt", task="detect")
result = yolo(source="YOLO11/ultralytics/assets/bus.jpg", save=True)