from src.frame_extractor import extract_frames
from src.embedder import generate_embeddings
from src.vector_store import create_index
from src.detector import detect_objects

import shutil
import os
import json

VIDEO_PATH = "videos/sample.mp4"
FRAME_FOLDER = "frames/sample/"

# Clean old frames
if os.path.exists(FRAME_FOLDER):
    shutil.rmtree(FRAME_FOLDER)

os.makedirs(FRAME_FOLDER, exist_ok=True)

print("🎬 Extracting frames...")
extract_frames(VIDEO_PATH, FRAME_FOLDER)

print("🧠 Generating embeddings...")
embeddings, paths = generate_embeddings(FRAME_FOLDER)

print("📦 Creating index...")
index = create_index(embeddings)

print("🔍 Running object detection...")
detections = detect_objects(FRAME_FOLDER)

with open("data/detections.json", "w") as f:
    json.dump(detections, f)

print("✅ Pipeline complete! Run UI now.")