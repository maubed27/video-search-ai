import streamlit as st
import json
import open_clip
import torch
from src.embedder import generate_embeddings
from src.vector_store import create_index
from src.query import search

FRAME_FOLDER = "frames/sample/"
VIDEO_PATH = "videos/sample.mp4"

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    model.to(device)
    return model

model = load_model()

@st.cache_resource
def load_system():
    embeddings, paths = generate_embeddings(FRAME_FOLDER)
    index = create_index(embeddings)

    with open("data/metadata.json", "r") as f:
        metadata = json.load(f)

    with open("data/detections.json", "r") as f:
        detection_map = json.load(f)

    time_map = {item["frame"]: item["timestamp"] for item in metadata}

    return index, paths, time_map, detection_map

index, paths, time_map, detection_map = load_system()

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

st.title("🎬 Intelligent Video Search Engine")
st.video(VIDEO_PATH)

query = st.text_input("🔍 Enter your query:")

if query:
    results = search(query, model, index, paths)

    query_words = query.lower().split()
    filtered_results = []

    for path, score in results:
        detected = detection_map.get(path, [])

        if any(word in detected for word in query_words):
            filtered_results.append((path, score))
        else:
            if score > 0.25:
                filtered_results.append((path, score))

    if len(filtered_results) == 0:
        st.warning("❌ No relevant results found")
    else:
        cols = st.columns(len(filtered_results))

        for i, (path, score) in enumerate(filtered_results):
            timestamp = time_map.get(path, 0)

            with cols[i]:
                st.image(path)
                st.caption(format_time(timestamp))
                st.caption(f"Score: {round(score, 3)}")
                st.caption(f"Objects: {', '.join(detection_map.get(path, []))}")
                st.video(VIDEO_PATH, start_time=int(timestamp))