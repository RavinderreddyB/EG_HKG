import json
import numpy as np
import torch
from pathlib import Path
from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

OUTPUT_PATH = Path("logs/kg_embeddings_report.json")

# Entities are prefixed by type (disease::, symptom::, drug::) so a disease
# and a drug that happen to share a name never collide in PyKEEN's shared
# head/tail vocabulary.
FETCH_TRIPLES_QUERY = """
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN 'disease::' + d.name AS head, 'has_symptom' AS relation, 'symptom::' + s.name AS tail
UNION
MATCH (dr:Drug)-[:TREATS]->(d:Disease)
RETURN 'drug::' + dr.name AS head, 'treats' AS relation, 'disease::' + d.name AS tail
"""

WRITE_SCORES_QUERY = """
UNWIND $rows AS row
MATCH (d:Disease {name: row.disease})-[r:HAS_SYMPTOM]->(s:Symptom {name: row.symptom})
SET r.structural_confidence = row.score
"""


def fetch_triples(driver):
    with driver.session() as session:
        rows = [r.data() for r in session.run(FETCH_TRIPLES_QUERY)]
    return np.array([[r["head"], r["relation"], r["tail"]] for r in rows])


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    triples = fetch_triples(driver)
    print(f"Total triples: {len(triples)}")

    tf = TriplesFactory.from_labeled_triples(triples)
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)

    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model="TransE",
        model_kwargs=dict(embedding_dim=32),
        training_kwargs=dict(num_epochs=200),
        random_seed=42,
        device="cpu",
    )

    metrics = {
        "hits@1": result.get_metric("hits@1"),
        "hits@3": result.get_metric("hits@3"),
        "hits@10": result.get_metric("hits@10"),
        "mean_reciprocal_rank": result.get_metric("mean_reciprocal_rank"),
    }
    print("\nLink prediction metrics (held-out test triples):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Score every HAS_SYMPTOM triple's structural plausibility, not just the
    # held-out test split -- this becomes the second, structure-based
    # confidence signal alongside the PubMed citation-based confidence.
    query = """
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
    RETURN d.name AS disease, s.name AS symptom
    """
    with driver.session() as session:
        pairs = [r.data() for r in session.run(query)]

    model = result.model
    factory = result.training
    scores = []
    rows = []
    for pair in pairs:
        head = f"disease::{pair['disease']}"
        tail = f"symptom::{pair['symptom']}"
        if head not in factory.entity_to_id or tail not in factory.entity_to_id:
            continue
        triple_id = torch.tensor([[
            factory.entity_to_id[head],
            factory.relation_to_id["has_symptom"],
            factory.entity_to_id[tail],
        ]], dtype=torch.long)
        score = float(model.score_hrt(triple_id).item())
        scores.append(score)
        rows.append({"disease": pair["disease"], "symptom": pair["symptom"], "raw_score": score})

    # min-max normalize raw TransE scores to [0, 1] for a second confidence signal
    lo, hi = min(scores), max(scores)
    for row in rows:
        row["score"] = (row["raw_score"] - lo) / (hi - lo) if hi > lo else 0.5

    with driver.session() as session:
        session.run(WRITE_SCORES_QUERY, rows=rows)

    driver.close()

    Path("logs").mkdir(exist_ok=True)
    report = {"metrics": metrics, "num_triples": len(triples), "num_scored_edges": len(rows)}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Wrote structural_confidence to {len(rows)} HAS_SYMPTOM edges")
    print(f"[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
