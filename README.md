# 🎬 Intelligent Video Search Engine

A semantic video search system that allows users to query video content using natural language and retrieve the most relevant moments with timestamps.

---

## 🚀 Overview

This project builds an AI-powered video retrieval system that:
- Processes videos into frames
- Extracts semantic embeddings using CLIP
- Performs fast similarity search using FAISS
- Enhances accuracy with object detection (YOLO)
- Provides an interactive UI for querying and viewing results

---

## 🧠 Architecture

### 1. Video Ingestion
- Input: Video file (`.mp4`)
- Frames extracted at **1 FPS** to balance performance and coverage

### 2. Frame Processing
- Frames resized for efficiency
- Each frame mapped to timestamp

### 3. Embedding Generation
- Uses **OpenCLIP (ViT-B/32)**
- Both image and text embeddings are **normalized**
- Enables semantic similarity matching

### 4. Vector Indexing
- FAISS (L2 index) used for fast nearest-neighbor search

### 5. Object Detection
- YOLOv8 used to detect objects in frames
- Helps filter results based on actual visual content

### 6. Query Processing
- Natural language query → CLIP embedding
- Retrieve top-K candidates from FAISS
- Re-rank using similarity score
- Filter using object detection (if applicable)

---

## 🎯 Features

- 🔍 Natural language video search
- ⏱ Timestamped results
- 🧠 Semantic understanding (CLIP)
- 🎯 Object-aware filtering (YOLO)
- 🎬 Video playback from exact moment
- ⚡ Fast retrieval using FAISS

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
pip install open_clip_torch torch torchvision faiss-cpu opencv-python pillow streamlit ultralytics