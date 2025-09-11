import cv2
from ultralytics import YOLO
import numpy as np

def calculate_average_confidence(boxes):
    """计算当前帧所有检测框的平均置信度"""
    if boxes is None or len(boxes) == 0:
        return 0.0
    confidences = [box.conf.cpu().numpy()[0] for box in boxes]
    return np.mean(confidences)

model_path = 'YOLOv11\\runs\\train\\train-200epoch-v11n.yaml\\weights\\best.pt'
model = YOLO(model_path)

#cap = cv2.VideoCapture(r"C:\Users\Lanyi\Desktop\Project\Eyetrack_Fursuit\images\eye_singal.mp4")
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

frame_count = 0
total_confidence = 0.0

try:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        cropped_frame = frame[:crop_height, :crop_width]

        results = model(cropped_frame)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                current_avg_confidence = calculate_average_confidence(boxes)
                total_confidence += current_avg_confidence
                cv2.putText(cropped_frame, 
                          f"Frame: {frame_count} | Avg Conf: {current_avg_confidence:.2f}",
                          (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, 
                          (0, 255, 255), 
                          2)

                for box in boxes:
                    confidence = box.conf.cpu().numpy()[0]
                    class_id = int(box.cls.cpu().numpy()[0])
                    bbox = box.xyxy.cpu().numpy()[0].astype(int)
                    
                    class_name = class_names.get(class_id, str(class_id))
                    
                    print(f"Frame {frame_count}: Class: {class_name}, Confidence: {confidence:.2f}, Bbox: {bbox}")

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


        if cv2.waitKey(1) == 27:
            print("ESC pressed, exiting...")
            break
        
        if frame_count == 173:
            break

finally:
    if frame_count > 0:
        overall_avg_confidence = total_confidence / frame_count
        print(f"\nProcessing summary:")
        print(f"Total frames processed: {frame_count}")
        print(f"Overall average confidence: {overall_avg_confidence:.4f}")

    # 确保资源被释放
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Resources released, program exited cleanly.")