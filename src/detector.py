from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

def detect_objects(frame_folder):
    detections = {}

    for file in os.listdir(frame_folder):
        path = os.path.join(frame_folder, file)

        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        results = model(path, verbose=False)

        labels = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                labels.append(label)

        detections[path] = labels

    return detections