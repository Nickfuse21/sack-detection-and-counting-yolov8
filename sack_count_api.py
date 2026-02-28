from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("video/Problem Statement Scenario3.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

out = cv2.VideoWriter("output_processed.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

count = 0
centroids = []
IGNORE_CLASSES = [0, 1, 2, 3, 5, 7] 

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    results = model(frame, imgsz=1280, conf=0.15, iou=0.45, verbose=False)[0]

    for box in results.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if cls in IGNORE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect_ratio = w / h

        if area > 50 and 0.3 < aspect_ratio < 3.0:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            new_obj = True
            for (px, py) in centroids:
                if np.linalg.norm([cx - px, cy - py]) < 25: 
                    new_obj = False
                    break
            
            if new_obj:
                count += 1
                centroids.append((cx, cy))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Sack {conf:.2f}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.rectangle(frame, (20, 20), (350, 80), (0,0,0), -1)
    cv2.putText(frame, f"Count: {count}", (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Total: {count}")