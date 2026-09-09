import json
import random
from pathlib import Path
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

RESULTS_PATH  = Path("logs/provenance_results.json")
BASELINE_PATH = Path("logs/baseline_metrics.json")
ENHANCED_PATH = Path("logs/enhanced_metrics.json")


def metric_coverage_rate(driver):
    total_q = "MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom) RETURN count(r) AS n"
    enriched_q = """
    MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
    WHERE r.evidence_src = 'PubMed'
    RETURN count(r) AS n
    """
    with driver.session() as session:
        total    = session.run(total_q).single()["n"]
        enriched = session.run(enriched_q).single()["n"]

    rate = enriched / total * 100 if total else 0
    return {"total_edges": total, "enriched_edges": enriched, "coverage_rate_pct": round(rate, 2)}


def metric_precision_at_50(driver, sample_size=50):
    query = """
    MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
    WHERE r.evidence_src = 'PubMed' AND r.pmid IS NOT NULL
    RETURN d.name AS disease, s.name AS symptom,
           r.pmid AS pmid, r.pub_title AS title,
           r.confidence AS confidence
    """
    with driver.session() as session:
        rows = [r.data() for r in session.run(query)]

    if not rows:
        return {"error": "No enriched edges found. Run add_provenance.py first."}

    sample = random.sample(rows, min(sample_size, len(rows)))

    print(f"\n--- Manual Annotation: Evidence Precision@{len(sample)} ---")
    print("For each entry, open https://pubmed.ncbi.nlm.nih.gov/<pmid>")
    print("Mark 1 if the paper is about that disease-symptom, 0 if not.\n")

    annotations = []
    for i, row in enumerate(sample):
        print(f"[{i+1}/{len(sample)}]")
        print(f"  Disease  : {row['disease']}")
        print(f"  Symptom  : {row['symptom']}")
        print(f"  PMID     : {row['pmid']}")
        print(f"  Title    : {row['title']}")
        print(f"  Confidence: {row['confidence']}")
        print(f"  URL      : https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}")

        while True:
            val = input("  Relevant? (1=yes / 0=no / s=skip): ").strip().lower()
            if val in ("1", "0", "s"):
                break
        if val != "s":
            annotations.append(int(val))
        print()

    if not annotations:
        return {"error": "No annotations recorded."}

    precision = sum(annotations) / len(annotations) * 100
    print(f"\nEvidence Precision@{len(annotations)}: {precision:.1f}%")
    return {
        "sample_size": len(annotations),
        "relevant_count": sum(annotations),
        "precision_pct": round(precision, 2),
    }


def metric_hallucination_reduction():
    if not BASELINE_PATH.exists() or not ENHANCED_PATH.exists():
        return {
            "error": (
                "Run scripts/run_full_evaluation.py twice:\n"
                "  1. Before enrichment -> logs/baseline_metrics.json\n"
                "  2. After enrichment  -> logs/enhanced_metrics.json"
            )
        }

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(ENHANCED_PATH) as f:
        enhanced = json.load(f)

    def avg(data, key):
        vals = [r[key] for r in data if key in r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    baseline_hall = avg(baseline, "hallucination")
    enhanced_hall = avg(enhanced, "hallucination")
    baseline_faith = avg(baseline, "faithfulness")
    enhanced_faith = avg(enhanced, "faithfulness")

    reduction = (baseline_hall - enhanced_hall) / baseline_hall * 100 if baseline_hall else 0

    return {
        "baseline_hallucination_rate": baseline_hall,
        "enhanced_hallucination_rate": enhanced_hall,
        "hallucination_reduction_pct": round(reduction, 2),
        "baseline_faithfulness": baseline_faith,
        "enhanced_faithfulness": enhanced_faith,
    }


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("=" * 55)
    print("PROVENANCE ENRICHMENT EVALUATION REPORT")
    print("=" * 55)

    print("\n[1] Coverage Rate")
    coverage = metric_coverage_rate(driver)
    for k, v in coverage.items():
        print(f"    {k}: {v}")

    print("\n[2] Evidence Precision@50 (manual annotation)")
    precision = metric_precision_at_50(driver)
    for k, v in precision.items():
        print(f"    {k}: {v}")

    print("\n[3] Downstream Hallucination Reduction")
    reduction = metric_hallucination_reduction()
    for k, v in reduction.items():
        print(f"    {k}: {v}")

    report = {
        "coverage": coverage,
        "precision_at_50": precision,
        "hallucination_reduction": reduction,
    }

    out_path = Path("logs/provenance_eval_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved to {out_path}")
    driver.close()


if __name__ == "__main__":
    main()
