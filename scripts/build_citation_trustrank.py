import json
from pathlib import Path
from neo4j import GraphDatabase
from kg.citation_graph import build_citation_graph, compute_trust_scores
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

OUTPUT_PATH = Path("logs/citation_trustrank.json")


def fetch_enriched_pmids(driver):
    query = """
    MATCH (:Disease)-[r:HAS_SYMPTOM]->(:Symptom)
    WHERE r.evidence_src = 'PubMed' AND r.pmid IS NOT NULL
    RETURN DISTINCT r.pmid AS pmid
    """
    with driver.session() as session:
        return [record["pmid"] for record in session.run(query)]


def write_trust_scores(driver, scores):
    rows = [{"pmid": pmid, "score": score} for pmid, score in scores.items()]
    query = """
    UNWIND $rows AS row
    MATCH (:Disease)-[r:HAS_SYMPTOM]->(:Symptom)
    WHERE r.pmid = row.pmid
    SET r.source_credibility = row.score
    """
    with driver.session() as session:
        session.run(query, rows=rows)


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    pmids = fetch_enriched_pmids(driver)
    print(f"Unique PubMed-linked PMIDs: {len(pmids)}")

    graph = build_citation_graph(pmids)
    print(f"Citation graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    scores = compute_trust_scores(graph)
    write_trust_scores(driver, scores)
    driver.close()

    Path("logs").mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"[OK] Wrote source_credibility to {len(scores)} PMIDs, saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
