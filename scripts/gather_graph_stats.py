import json
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
with driver.session() as session:
    struct_conf = [
        r["v"] for r in session.run(
            "MATCH ()-[r:HAS_SYMPTOM]->() WHERE r.structural_confidence IS NOT NULL "
            "RETURN r.structural_confidence AS v"
        )
    ]
    total_edges = session.run("MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom) RETURN count(r) AS n").single()["n"]
    enriched = session.run(
        "MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom) WHERE r.evidence_src='PubMed' AND r.pmid IS NOT NULL RETURN count(r) AS n"
    ).single()["n"]
    n_diseases = session.run("MATCH (d:Disease) RETURN count(d) AS n").single()["n"]
    n_symptoms = session.run("MATCH (s:Symptom) RETURN count(s) AS n").single()["n"]
    n_drugs = session.run("MATCH (d:Drug) RETURN count(d) AS n").single()["n"]
    n_treats = session.run("MATCH ()-[r:TREATS]->() RETURN count(r) AS n").single()["n"]
    n_interacts = session.run("MATCH ()-[r:INTERACTS_WITH]->() RETURN count(r) AS n").single()["n"]
    n_cooccurs = session.run("MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) AS n").single()["n"]
driver.close()

print(f"structural_confidence values: {len(struct_conf)}")
print(f"total HAS_SYMPTOM edges: {total_edges}, enriched: {enriched}")
print(f"diseases: {n_diseases}, symptoms: {n_symptoms}, drugs: {n_drugs}")
print(f"TREATS: {n_treats}, INTERACTS_WITH: {n_interacts}, CO_OCCURS_WITH: {n_cooccurs}")

with open("logs/graph_stats.json", "w") as f:
    json.dump({
        "structural_confidence": struct_conf,
        "total_edges": total_edges,
        "enriched_edges": enriched,
        "n_diseases": n_diseases,
        "n_symptoms": n_symptoms,
        "n_drugs": n_drugs,
        "n_treats": n_treats,
        "n_interacts": n_interacts,
        "n_cooccurs": n_cooccurs,
    }, f, indent=2)
print("Saved logs/graph_stats.json")
