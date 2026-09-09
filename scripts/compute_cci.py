"""Confidence Consistency Index (CCI) -- how much do the 3 independent
confidence signals (semantic, source-credibility/PageRank, structural/TransE)
agree with each other, per edge, across the enriched HAS_SYMPTOM edges.

Run from the project root: python -m scripts.compute_cci
"""
import json
import statistics
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

OUTPUT_PATH = "logs/confidence_consistency_index.json"


def minmax_normalize(values):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    query = """
    MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
    WHERE r.evidence_src = 'PubMed'
      AND r.confidence IS NOT NULL
      AND r.source_credibility IS NOT NULL
      AND r.structural_confidence IS NOT NULL
    RETURN d.name AS disease, s.name AS symptom, d.source AS source,
           r.confidence AS semantic,
           r.source_credibility AS credibility,
           r.structural_confidence AS structural
    """

    with driver.session() as session:
        rows = [r.data() for r in session.run(query)]
    driver.close()

    print(f"Edges with all 3 confidence signals: {len(rows)}")

    semantic_raw = [r["semantic"] for r in rows]
    credibility_raw = [r["credibility"] for r in rows]
    structural_raw = [r["structural"] for r in rows]

    semantic_n = minmax_normalize(semantic_raw)
    credibility_n = minmax_normalize(credibility_raw)
    structural_n = minmax_normalize(structural_raw)

    per_edge = []
    for i, r in enumerate(rows):
        vals = [semantic_n[i], credibility_n[i], structural_n[i]]
        std = statistics.pstdev(vals)
        consistency = 1 - std
        per_edge.append({
            "disease": r["disease"], "symptom": r["symptom"], "source": r["source"],
            "semantic_norm": vals[0], "credibility_norm": vals[1], "structural_norm": vals[2],
            "consistency": consistency,
        })

    cci = sum(e["consistency"] for e in per_edge) / len(per_edge)

    corr_sem_cred = pearson(semantic_n, credibility_n)
    corr_sem_struct = pearson(semantic_n, structural_n)
    corr_cred_struct = pearson(credibility_n, structural_n)

    print(f"\nCCI (mean per-edge agreement across 3 normalized signals, 1.0 = perfect): {cci:.4f}")
    print("Pairwise correlations (population-level):")
    print(f"  semantic <-> source credibility : {corr_sem_cred:.4f}")
    print(f"  semantic <-> structural         : {corr_sem_struct:.4f}")
    print(f"  source credibility <-> structural: {corr_cred_struct:.4f}")

    by_source = {}
    for e in per_edge:
        by_source.setdefault(e["source"] or "unknown", []).append(e["consistency"])

    print("\nBy disease source:")
    for src, vals in by_source.items():
        print(f"  {src:20s} n={len(vals):4d}  CCI={sum(vals) / len(vals):.4f}")

    report = {
        "n_edges": len(rows),
        "cci": cci,
        "pairwise_correlations": {
            "semantic_credibility": corr_sem_cred,
            "semantic_structural": corr_sem_struct,
            "credibility_structural": corr_cred_struct,
        },
        "by_source": {src: {"n": len(vals), "cci": sum(vals) / len(vals)} for src, vals in by_source.items()},
        "raw_signal_ranges": {
            "semantic": [min(semantic_raw), max(semantic_raw)],
            "source_credibility": [min(credibility_raw), max(credibility_raw)],
            "structural": [min(structural_raw), max(structural_raw)],
        },
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
