import torch
import open_clip
from PIL import Image
import os
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model.to(device)


def generate_embeddings(frame_folder):
    embeddings = []
    paths = []

    for file in os.listdir(frame_folder):
        path = os.path.join(frame_folder, file)

        if not os.path.isfile(path):
            continue

        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image = preprocess(Image.open(path)).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(image)

        # Normalize
        emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb.cpu().numpy())
        paths.append(path)

    if len(embeddings) == 0:
        raise ValueError("❌ No valid frames found")

    return np.vstack(embeddings), paths