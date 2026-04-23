import cv2
import os
import json

def extract_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(fps / frame_rate)

    count = 0
    saved = 0
    metadata = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            time_sec = count / fps

            frame = cv2.resize(frame, (320, 240))
            frame_path = os.path.join(output_folder, f"frame_{saved}.jpg")

            cv2.imwrite(frame_path, frame)

            metadata.append({
                "frame": frame_path,
                "timestamp": time_sec
            })

            saved += 1

        count += 1

    cap.release()

    os.makedirs("data", exist_ok=True)

    with open("data/metadata.json", "w") as f:
        json.dump(metadata, f)

    print(f"✅ Extracted {saved} frames")