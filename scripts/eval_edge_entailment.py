"""Edge-level entailment check for the "evidence-grounded" claim (Section 3.3).

The citation-selection pipeline (kg/pubmed_linker.py) picks a PubMed paper
via lexical co-occurrence (disease/symptom terms in title/abstract, recency,
non-empty journal field) and scores confidence via embedding cosine
similarity -- neither step checks whether the abstract actually entails the
specific disease-symptom claim, only that it's topically related. This
script closes that gap by running the same HHEM entailment model already
used for generated-answer faithfulness (Section 3.6, evaluation/faithfulness_nli.py)
against every enriched edge: premise = the cited PubMed abstract,
hypothesis = "{disease} is associated with {symptom}."

Run over the full 770-edge population, not a sample -- same cost order as
the embedding-model ablation (scripts/rerun_embedding_ablation_full.py),
which already re-fetches and re-encodes all 770 edges in a few minutes.

Run from the project root: python -m scripts.eval_edge_entailment
"""
import json
import os
import re
import time

import requests
from neo4j import GraphDatabase

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from evaluation.faithfulness_nli import FaithfulnessScorer, LOW_CONSISTENCY_THRESHOLD, MAX_EVIDENCE_PIECE_CHARS

OUTPUT_PATH = "logs/edge_entailment_check.json"

PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_API_KEY = os.getenv("NCBI_API_KEY", None)
if NCBI_API_KEY == "your_ncbi_key":
    NCBI_API_KEY = None
REQUEST_DELAY = 0.11 if NCBI_API_KEY else 0.35
BATCH_SIZE = 150


def fetch_edges():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        rows = session.run("""
            MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
            WHERE r.evidence_src = 'PubMed' AND r.pmid IS NOT NULL
            RETURN d.name AS disease, s.name AS symptom, d.source AS source, r.pmid AS pmid
        """)
        edges = [rec.data() for rec in rows]
    driver.close()
    return edges


def fetch_abstracts_batch(pmids):
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    try:
        r = requests.get(PUBMED_FETCH_URL, params=params, timeout=30)
        r.raise_for_status()
        xml = r.text
        abstracts = {}
        articles = re.split(r"<PubmedArticle>", xml)[1:]
        for article in articles:
            pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
            abstract_match = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
            if pmid_match:
                pmid = pmid_match.group(1)
                abstracts[pmid] = " ".join(abstract_match).strip() if abstract_match else ""
        return abstracts
    except Exception as e:
        print(f"  [WARN] batch fetch failed: {e}")
        return {}


def main():
    edges = fetch_edges()
    print(f"Total enriched edges with a PMID: {len(edges)}")

    unique_pmids = sorted({e["pmid"] for e in edges})
    print(f"Unique PMIDs to fetch: {len(unique_pmids)}")

    abstracts = {}
    for i in range(0, len(unique_pmids), BATCH_SIZE):
        batch = unique_pmids[i:i + BATCH_SIZE]
        abstracts.update(fetch_abstracts_batch(batch))
        print(f"  fetched {min(i + BATCH_SIZE, len(unique_pmids))}/{len(unique_pmids)} PMIDs")
        time.sleep(REQUEST_DELAY)

    edges_with_abstract = [e for e in edges if abstracts.get(e["pmid"])]
    missing = len(edges) - len(edges_with_abstract)
    print(f"Edges with no fetchable abstract (excluded): {missing}")

    print("Loading HHEM entailment model...")
    scorer = FaithfulnessScorer()

    results = []
    for i, e in enumerate(edges_with_abstract):
        abstract = abstracts[e["pmid"]][:MAX_EVIDENCE_PIECE_CHARS]
        hypothesis = f"{e['disease']} is associated with {e['symptom']}."
        score = scorer.score_pair(abstract, hypothesis)
        results.append({
            "disease": e["disease"], "symptom": e["symptom"], "source": e["source"],
            "pmid": e["pmid"], "score": score,
            "entailed": score >= LOW_CONSISTENCY_THRESHOLD,
        })
        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(edges_with_abstract)}")

    n = len(results)
    entailed = sum(1 for r in results if r["entailed"])
    mean_score = sum(r["score"] for r in results) / n

    print(f"\nn = {n}")
    print(f"Mean entailment score: {mean_score:.4f}")
    print(f"Entailed (score >= {LOW_CONSISTENCY_THRESHOLD}): {entailed}/{n} ({entailed/n*100:.1f}%)")

    by_source = {}
    for r in results:
        by_source.setdefault(r["source"], []).append(r)
    print("\nBy source:")
    for src, rs in by_source.items():
        e_rate = sum(1 for r in rs if r["entailed"]) / len(rs)
        print(f"  {src:20s} n={len(rs):4d}  entailed={e_rate*100:.1f}%  mean_score={sum(r['score'] for r in rs)/len(rs):.4f}")

    report = {
        "n": n,
        "threshold": LOW_CONSISTENCY_THRESHOLD,
        "mean_score": mean_score,
        "entailed_count": entailed,
        "entailed_rate": entailed / n,
        "by_source": {
            src: {
                "n": len(rs),
                "entailed_rate": sum(1 for r in rs if r["entailed"]) / len(rs),
                "mean_score": sum(r["score"] for r in rs) / len(rs),
            } for src, rs in by_source.items()
        },
        "results": results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
