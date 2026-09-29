"""Compares TransE against DistMult, ComplEx, and RotatE on the same
link-prediction task (Section 7.6.1's self-identified limitation: no
alternative KG-embedding method was tried against TransE). Same triples,
same 80/10/10 split, same seed as scripts/build_kg_embeddings.py -- this
is read-only and does not write structural_confidence back to Neo4j;
TransE remains the production signal.

Run from the project root: python -m scripts.compare_kg_embedding_methods
"""
import json
import numpy as np
from pathlib import Path
from neo4j import GraphDatabase
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

OUTPUT_PATH = Path("logs/kg_embedding_method_comparison.json")

FETCH_TRIPLES_QUERY = """
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN 'disease::' + d.name AS head, 'has_symptom' AS relation, 'symptom::' + s.name AS tail
UNION
MATCH (dr:Drug)-[:TREATS]->(d:Disease)
RETURN 'drug::' + dr.name AS head, 'treats' AS relation, 'disease::' + d.name AS tail
"""

MODELS = ["TransE", "DistMult", "ComplEx", "RotatE"]


def fetch_triples(driver):
    with driver.session() as session:
        rows = [r.data() for r in session.run(FETCH_TRIPLES_QUERY)]
    return np.array([[r["head"], r["relation"], r["tail"]] for r in rows])


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    triples = fetch_triples(driver)
    driver.close()
    print(f"Total triples: {len(triples)}")

    tf = TriplesFactory.from_labeled_triples(triples)
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)

    results = {}
    for model_name in MODELS:
        print(f"\n=== Training {model_name} ===")
        result = pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model=model_name,
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
        results[model_name] = metrics
        print(f"  Hits@1={metrics['hits@1']:.4f}  Hits@3={metrics['hits@3']:.4f}  "
              f"Hits@10={metrics['hits@10']:.4f}  MRR={metrics['mean_reciprocal_rank']:.4f}")

    print("\n=== Summary ===")
    print(f"{'Model':10s} {'Hits@1':>8s} {'Hits@3':>8s} {'Hits@10':>8s} {'MRR':>8s}")
    for model_name, m in results.items():
        print(f"{model_name:10s} {m['hits@1']:8.4f} {m['hits@3']:8.4f} {m['hits@10']:8.4f} {m['mean_reciprocal_rank']:8.4f}")

    report = {"num_triples": len(triples), "results": results}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
