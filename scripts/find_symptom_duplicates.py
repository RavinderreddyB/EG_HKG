"""Read-only scan for likely-duplicate symptom entities (entity-alignment
gap, see HANDOFF.md / paper discussion). Does NOT modify the graph --
embeds every symptom name with the same PubMedBERT model already used
elsewhere in this project, computes pairwise cosine similarity, and
reports the highest-similarity pairs as merge candidates for manual review.

Run from the project root: python -m scripts.find_symptom_duplicates
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDING_MODEL

OUTPUT_PATH = "logs/symptom_duplicate_candidates.json"
TOP_N = 150


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        names = sorted(r["name"] for r in s.run("MATCH (s:Symptom) RETURN s.name AS name"))
    driver.close()

    print(f"Symptom nodes: {len(names)}")
    print(f"Loading embedding model: {EMBEDDING_MODEL}", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(names, show_progress_bar=True, normalize_embeddings=True)

    sim_matrix = embeddings @ embeddings.T

    pairs = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(sim_matrix[i, j]), names[i], names[j]))

    pairs.sort(key=lambda x: -x[0])

    print(f"\nTop {TOP_N} candidate duplicate pairs (highest similarity first):\n")
    top = pairs[:TOP_N]
    for score, a, b in top:
        print(f"  {score:.4f}  {a!r} <-> {b!r}")

    report = {
        "n_symptoms": n,
        "n_pairs_scored": len(pairs),
        "top_candidates": [{"similarity": s, "a": a, "b": b} for s, a, b in top],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
