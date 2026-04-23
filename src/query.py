import open_clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def search(query, model, index, paths, top_k=20, threshold=0.2):
    text = open_clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    query_vec = text_features.cpu().numpy()

    distances, indices = index.search(query_vec, top_k)

    results = []

    for i in range(top_k):
        path = paths[indices[0][i]]
        similarity = 1 - distances[0][i]
        results.append((path, similarity))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    filtered = [r for r in results if r[1] > threshold]

    if len(filtered) == 0:
        return results[:5]

    return filtered[:5]