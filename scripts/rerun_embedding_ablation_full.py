"""Re-run of the embedding-model ablation (Section 6.1) over the FULL 770
enriched edges, not just the 224-edge subset used previously.

The earlier ablation (data/processed/embedding_ablation_224_edges.json) was
scoped to the 224 edges sourced from the original DiseaseAndSymptoms
dataset only. Direct inspection showed the other 546 MedlinePlus-sourced
enriched edges carry real embedding-based `confidence` values from the same
PubMedLinker pipeline (identical distribution signature: 0.5 floor,
continuous spread, pmid + relevance_score populated) -- so there was no
methodological reason to exclude them. This script closes that gap.

Domain-specific side: the real stored per-edge `confidence` property
(PubMedBERT, already computed by kg/pubmed_linker.py during enrichment).

Generic side: replicates kg/pubmed_linker.py's _compute_confidence()
methodology exactly -- full abstract text fetched live via NCBI efetch,
truncated to 512 characters, encoded with all-MiniLM-L6-v2. Edges whose
PMID returns no abstract get the same 0.5 fallback used in the original
pipeline.

Run from the project root: python -m scripts.rerun_embedding_ablation_full
"""
import json
import os
import re
import time

import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_API_KEY = os.getenv("NCBI_API_KEY", None)
REQUEST_DELAY = 0.11 if NCBI_API_KEY else 0.35
BATCH_SIZE = 150

GENERIC_MODEL = "all-MiniLM-L6-v2"
OUTPUT_PATH = "data/processed/embedding_ablation_770_edges.json"


def fetch_edges():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        rows = session.run("""
            MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
            WHERE r.evidence_src = 'PubMed' AND r.confidence IS NOT NULL AND r.pmid IS NOT NULL
            RETURN d.name AS disease, s.name AS symptom, r.pmid AS pmid, r.confidence AS confidence
        """)
        edges = [rec.data() for rec in rows]
    driver.close()
    return edges


def fetch_abstracts_batch(pmids):
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
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
                abstract_text = " ".join(abstract_match).strip() if abstract_match else ""
                abstracts[pmid] = abstract_text
        return abstracts
    except Exception as e:
        print(f"  [WARN] batch fetch failed: {e}")
        return {}


def main():
    edges = fetch_edges()
    print(f"Total enriched edges: {len(edges)}")

    unique_pmids = sorted({e["pmid"] for e in edges})
    print(f"Unique PMIDs to fetch: {len(unique_pmids)}")

    abstracts = {}
    for i in range(0, len(unique_pmids), BATCH_SIZE):
        batch = unique_pmids[i:i + BATCH_SIZE]
        abstracts.update(fetch_abstracts_batch(batch))
        print(f"  fetched {min(i + BATCH_SIZE, len(unique_pmids))}/{len(unique_pmids)} PMIDs")
        time.sleep(REQUEST_DELAY)

    missing = [p for p in unique_pmids if not abstracts.get(p)]
    print(f"PMIDs with no abstract text (will fall back to 0.5): {len(missing)}")

    print(f"Loading generic model: {GENERIC_MODEL}")
    model = SentenceTransformer(GENERIC_MODEL)

    generic_scores = []
    domain_specific_scores = []
    for e in edges:
        abstract = abstracts.get(e["pmid"], "")
        if not abstract:
            generic_conf = 0.50
        else:
            query_vec = model.encode([f"{e['disease']} {e['symptom']}"])
            abstract_vec = model.encode([abstract[:512]])
            sim = cosine_similarity(query_vec, abstract_vec)[0][0]
            generic_conf = round(float(sim), 4)

        generic_scores.append(generic_conf)
        domain_specific_scores.append(e["confidence"])

    out = {
        "n": len(edges),
        "generic_model": GENERIC_MODEL,
        "domain_specific_model": "PubMedBERT (pritamdeka/S-PubMedBert-MS-MARCO)",
        "generic": generic_scores,
        "domain_specific": domain_specific_scores,
        "pmids_missing_abstract": len(missing),
        "unique_pmids": len(unique_pmids),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[OK] Wrote {OUTPUT_PATH}")
    print(f"Generic mean:         {sum(generic_scores)/len(generic_scores):.4f}, min={min(generic_scores):.4f}")
    print(f"Domain-specific mean: {sum(domain_specific_scores)/len(domain_specific_scores):.4f}, min={min(domain_specific_scores):.4f}")


if __name__ == "__main__":
    main()
