import cv2
from ultralytics import YOLO

model = YOLO('models/yolo11n.pt')
cap = cv2.VideoCpature(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        result = model(frame)

        annotated_frame = result[0].plot()

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()