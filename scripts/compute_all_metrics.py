"""Single entry point that consolidates every metric reported in
OBJECTIVE_I_RESEARCH_PAPER.md into one JSON file and one printed summary.

Cheap, live-queryable metrics (graph stats, provenance coverage, confidence
signal summary, CCI) are recomputed fresh every run. Expensive ones --
LLM-based (faithfulness/hallucination), training runs (TransE embeddings),
or the 20-seed held-out evaluation (Hit@K/MRR/ARDA, ~2 min) -- are read
from their existing saved report files instead of recomputed inline, since
they have dedicated scripts (see the runbook below). ARDA/Hit@K used to be
computed inline here with a single seed=42 -- removed after that single-run
approach was found to disagree with Table 6's own separate single run
(96.80% vs 94.40% Hit@1 on the same graph); both are superseded by
scripts.compute_holdout_eval's 20-seed mean+-SD evaluation.

Runbook to produce everything this script reads, in order:
  python -m scripts.build_kg
  python -m scripts.build_drug_kg
  python -m scripts.build_interactions
  python -m scripts.add_provenance
  python -m scripts.build_citation_trustrank
  python -m scripts.build_kg_embeddings          # -> logs/kg_embeddings_report.json
  python -m scripts.compute_holdout_eval         # -> logs/holdout_eval_multiseed_report.json
  python -m scripts.generate_batch_runs 34 777
  python -m scripts.eval_faithfulness_final      # -> logs/faithfulness_nli_final_report.json
  python -m scripts.compute_all_metrics          # this script, run last

Run from the project root: python -m scripts.compute_all_metrics
"""
import json
import statistics
from pathlib import Path
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

OUTPUT_PATH = Path("logs/all_metrics_summary.json")

KG_EMBEDDINGS_REPORT = Path("logs/kg_embeddings_report.json")
FAITHFULNESS_REPORT = Path("logs/faithfulness_nli_final_report.json")
HOLDOUT_EVAL_REPORT = Path("logs/holdout_eval_multiseed_report.json")


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
    return num / (da * db) if da and db else 0.0


def graph_stats(driver):
    with driver.session() as session:
        counts = session.run("""
            MATCH (d:Disease) WITH count(d) AS n_diseases
            MATCH (s:Symptom) WITH n_diseases, count(s) AS n_symptoms
            MATCH ()-[r:HAS_SYMPTOM]->() WITH n_diseases, n_symptoms, count(r) AS n_has_symptom
            MATCH ()-[r2:TREATS]->() WITH n_diseases, n_symptoms, n_has_symptom, count(r2) AS n_treats
            MATCH ()-[r3:HAS_TREATMENT]->() WITH n_diseases, n_symptoms, n_has_symptom, n_treats, count(r3) AS n_has_treatment
            MATCH ()-[r4:INTERACTS_WITH]->() WITH n_diseases, n_symptoms, n_has_symptom, n_treats, n_has_treatment, count(r4) AS n_interacts
            RETURN n_diseases, n_symptoms, n_has_symptom, n_treats, n_has_treatment, n_interacts
        """).single()

        by_source = session.run("""
            MATCH (d:Disease)
            RETURN d.source AS source, count(d) AS n_diseases
        """)
        by_source_diseases = {r["source"] or "unknown": r["n_diseases"] for r in by_source}

    return {"totals": dict(counts), "diseases_by_source": by_source_diseases}


def provenance_coverage(driver):
    with driver.session() as session:
        rows = session.run("""
            MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
            RETURN d.source AS source, r.evidence_src AS evidence_src, count(*) AS n
        """)
        data = [r.data() for r in rows]

    by_source = {}
    for row in data:
        src = row["source"] or "unknown"
        by_source.setdefault(src, {"total": 0, "pubmed": 0})
        by_source[src]["total"] += row["n"]
        if row["evidence_src"] == "PubMed":
            by_source[src]["pubmed"] += row["n"]

    total = sum(v["total"] for v in by_source.values())
    pubmed = sum(v["pubmed"] for v in by_source.values())

    result = {
        "overall": {"total": total, "pubmed": pubmed, "coverage": pubmed / total if total else 0.0},
        "by_source": {
            src: {**v, "coverage": v["pubmed"] / v["total"] if v["total"] else 0.0}
            for src, v in by_source.items()
        },
    }
    return result


def confidence_signals(driver):
    with driver.session() as session:
        rows = session.run("""
            MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
            WHERE r.evidence_src = 'PubMed'
            RETURN r.confidence AS semantic, r.source_credibility AS credibility,
                   r.structural_confidence AS structural
        """)
        data = [r.data() for r in rows]

    semantic = [r["semantic"] for r in data if r["semantic"] is not None]
    credibility = [r["credibility"] for r in data if r["credibility"] is not None]
    structural = [r["structural"] for r in data if r["structural"] is not None]

    def summary(vals):
        if not vals:
            return None
        return {
            "n": len(vals), "mean": statistics.mean(vals), "min": min(vals), "max": max(vals),
            "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }

    return {"semantic": summary(semantic), "source_credibility": summary(credibility), "structural": summary(structural)}


def compute_cci(driver):
    with driver.session() as session:
        rows = session.run("""
            MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
            WHERE r.evidence_src = 'PubMed'
              AND r.confidence IS NOT NULL AND r.source_credibility IS NOT NULL AND r.structural_confidence IS NOT NULL
            RETURN d.source AS source, r.confidence AS semantic, r.source_credibility AS credibility,
                   r.structural_confidence AS structural
        """)
        data = [r.data() for r in rows]

    if not data:
        return None

    semantic_n = minmax_normalize([r["semantic"] for r in data])
    credibility_n = minmax_normalize([r["credibility"] for r in data])
    structural_n = minmax_normalize([r["structural"] for r in data])

    per_edge_consistency = []
    by_source = {}
    for i, r in enumerate(data):
        vals = [semantic_n[i], credibility_n[i], structural_n[i]]
        consistency = 1 - statistics.pstdev(vals)
        per_edge_consistency.append(consistency)
        by_source.setdefault(r["source"] or "unknown", []).append(consistency)

    return {
        "n_edges": len(data),
        "cci": sum(per_edge_consistency) / len(per_edge_consistency),
        "pairwise_correlations": {
            "semantic_credibility": pearson(semantic_n, credibility_n),
            "semantic_structural": pearson(semantic_n, structural_n),
            "credibility_structural": pearson(credibility_n, structural_n),
        },
        "by_source": {src: {"n": len(v), "cci": sum(v) / len(v)} for src, v in by_source.items()},
    }


def load_if_exists(path, label):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    print(f"  [MISSING] {label}: {path} not found -- run its script first (see this file's docstring)")
    return None


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("Computing live graph metrics...", flush=True)
    summary = {
        "graph_stats": graph_stats(driver),
        "provenance_coverage": provenance_coverage(driver),
        "confidence_signals": confidence_signals(driver),
        "cci": compute_cci(driver),
    }
    driver.close()

    print("Loading pre-computed reports (TransE embeddings, held-out eval/ARDA, faithfulness)...", flush=True)
    summary["transe_embeddings"] = load_if_exists(KG_EMBEDDINGS_REPORT, "TransE embeddings report")
    summary["holdout_eval"] = load_if_exists(HOLDOUT_EVAL_REPORT, "20-seed held-out eval / ARDA report")
    summary["faithfulness"] = load_if_exists(FAITHFULNESS_REPORT, "Faithfulness/hallucination report")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[OK] Saved consolidated metrics to {OUTPUT_PATH}\n")

    print("=== SUMMARY ===")
    gs = summary["graph_stats"]["totals"]
    print(f"Graph: {gs['n_diseases']} diseases, {gs['n_symptoms']} symptoms, "
          f"{gs['n_has_symptom']} HAS_SYMPTOM, {gs['n_treats']} TREATS, "
          f"{gs['n_has_treatment']} HAS_TREATMENT, {gs['n_interacts']} INTERACTS_WITH")

    pc = summary["provenance_coverage"]["overall"]
    print(f"Provenance coverage: {pc['pubmed']}/{pc['total']} ({pc['coverage']:.2%})")

    cs = summary["confidence_signals"]
    if cs["semantic"]:
        print(f"Semantic confidence: mean {cs['semantic']['mean']:.4f} (n={cs['semantic']['n']})")
    if cs["source_credibility"]:
        print(f"Source credibility: mean {cs['source_credibility']['mean']:.5f} (n={cs['source_credibility']['n']})")
    if cs["structural"]:
        print(f"Structural confidence: mean {cs['structural']['mean']:.4f}, std {cs['structural']['stdev']:.4f} (n={cs['structural']['n']})")

    if summary["cci"]:
        print(f"CCI: {summary['cci']['cci']:.4f} (n={summary['cci']['n_edges']})")

    if summary["holdout_eval"]:
        h = summary["holdout_eval"]["summary_mean_sd"]
        print(f"Held-out eval ({summary['holdout_eval']['n_seeds']} seeds): "
              f"Hit@1 {h['hit1']['mean']:.2%} +/- {h['hit1']['sd']:.2%}, "
              f"Hit@3 {h['hit3']['mean']:.2%} +/- {h['hit3']['sd']:.2%}, "
              f"MRR {h['mrr']['mean']:.4f} +/- {h['mrr']['sd']:.4f}")
        print(f"ARDA (same seeds): accuracy {h['arda_accuracy']['mean']:.4f} +/- {h['arda_accuracy']['sd']:.4f}, "
              f"precision {h['precision']['mean']:.4f} +/- {h['precision']['sd']:.4f}, "
              f"recall {h['recall']['mean']:.4f} +/- {h['recall']['sd']:.4f}")

    if summary["faithfulness"]:
        fr = summary["faithfulness"]
        print(f"Faithfulness: {fr['avg_consistency']:.4f}, hallucination rate: {fr['avg_low_consistency_rate']:.4f} (n={fr['num_scored']})")

    if summary["transe_embeddings"]:
        print("TransE embeddings report: loaded (see logs/kg_embeddings_report.json for detail)")


if __name__ == "__main__":
    main()
