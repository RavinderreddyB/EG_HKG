import os
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config.settings import EMBEDDING_MODEL

torch.set_num_threads(os.cpu_count())

INPUT = "data/raw/medquad/medquad.jsonl"
INDEX_PATH = "data/vector_store/medquad.faiss"
MAP_PATH = "data/vector_store/medquad_map.json"

model = SentenceTransformer(EMBEDDING_MODEL)
model.max_seq_length = 256  # MedQuAD answers rarely need more; cuts CPU attention cost ~3-4x

texts = []
mapping = []

with open(INPUT, "r", encoding="utf8") as f:
    for line in f:
        item = json.loads(line)

        text = f"""
{item['answer']}
"""

        texts.append(text)
        mapping.append(item)

print("Encoding...")

embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(MAP_PATH, "w") as f:
    json.dump(mapping, f)

print("[OK] Vector DB built")