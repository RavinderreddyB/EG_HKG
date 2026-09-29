import faiss
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
from config.settings import EMBEDDING_MODEL, RERANKER_MODEL

# ensure directory exists
Path("data/vector_store").mkdir(parents=True, exist_ok=True)


class MedicalRAG:

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # load FAISS index
        self.index = faiss.read_index("data/vector_store/medquad.faiss")

        # load metadata
        with open("data/vector_store/medquad_map.json", "r", encoding="utf8") as f:
            self.mapping = json.load(f)

    def get_candidates(self, query, disease=None, top_k=20):
        """FAISS + cross-encoder scored candidates, before the final top-5 cutoff.
        Exposed separately so scripts/train_trust_rank.py can build a labeled
        training set from the full candidate pool, not just the survivors."""

        # encode query
        vec = self.model.encode([query]).astype("float32")

        # search FAISS
        distances, indices = self.index.search(vec, top_k)

        candidates = []

        # normalize disease
        disease = disease.lower() if disease else None

        for i, d in zip(indices[0], distances[0]):

            item = self.mapping[i]

            question = item.get("question", "").lower()
            answer = item.get("answer")

            # skip bad data
            if not answer:
                continue

            answer_lower = answer.lower()

            # disease-aware filtering (balanced, not too strict)
            if disease:
                if disease not in question and disease not in answer_lower:
                    if len(candidates) >= 2:
                        continue

            candidates.append({
                "text": answer,
                "question": item.get("question", ""),
                "score": float(1 / (1 + d))
            })

        if not candidates:
            return []

        # cross-encoder reranking: scores true (query, answer) relevance
        # directly, replacing the old bag-of-treatment-keywords filter
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for c, score in zip(candidates, rerank_scores):
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        return candidates

    def retrieve(self, query, disease=None, top_k=20):
        return self.get_candidates(query, disease=disease, top_k=top_k)[:5]