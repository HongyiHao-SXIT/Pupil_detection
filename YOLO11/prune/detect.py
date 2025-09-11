import cv2
from ultralytics import YOLO

model_path = 'YOLOv11\\runs\\train\\train-200epoch-v11n.yaml\\weights\\best.pt'
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening camera")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

crop_width = width
crop_height = height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('runs/predict2/eye_data_detected.mp4', fourcc, fps, (crop_width, crop_height))

class_names = {0: "eyeball", 1: "pupil"}
confidence_threshold = 0.5

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cropped_frame = frame[:crop_height, :crop_width]

        results = model(cropped_frame)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = box.conf.cpu().numpy()[0]
                    if confidence >= confidence_threshold:
                        class_id = int(box.cls.cpu().numpy()[0])
                        bbox = box.xyxy.cpu().numpy()[0].astype(int)
                        
                        class_name = class_names.get(class_id, str(class_id))
                        
                        print(f"Class: {class_name}, Confidence: {confidence:.2f}, Bbox: {bbox}")

                        cv2.rectangle(cropped_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(cropped_frame, 
                                   f"{class_name}: {confidence:.2f}",
                                   (bbox[0], bbox[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.9, 
                                   (0, 255, 0), 
                                   2)

        cv2.imshow('YOLOv11 Eye Detection', cropped_frame)

        out.write(cropped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()