"""Partial external validation against Rotmensch et al.'s physician-derived
health knowledge graph (Section 9.4) -- run now rather than deferred, since
the data is public and the comparison is feasible without new infrastructure.

Source: Rotmensch, Halpern, Tlimat, Horng, Sontag, "Learning a Health
Knowledge Graph from Electronic Medical Records," Sci Rep 7 (2017) 5994.
Data: https://github.com/clinicalml/HealthKnowledgeGraph
(157 diseases, 491 symptoms, learned from >270,000 patients' EMR data via a
noisy-OR Bayesian network -- an independent, EMR-derived source, not
self-referential to this project's graph.)

Method: for each disease present in both graphs, check what fraction of
this project's HAS_SYMPTOM edges for that disease are corroborated (the
same symptom, or a close textual match) by Rotmensch et al.'s graph. This
is a lower bound -- symptom vocabulary differs between a patient-facing
checklist and EMR-derived terminology (the same vocabulary-mismatch effect
already documented in Section 7.3), so near-miss phrasing will undercount
true agreement.

Run from the project root: python -m scripts.eval_external_validation
"""
import csv
import json
import re

from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

EXTERNAL_PATH = "data/raw/external/rotmensch_health_kg.csv"
OUTPUT_PATH = "logs/external_validation_rotmensch.json"


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[_\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def load_external():
    external = {}
    with open(EXTERNAL_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            disease, symptoms_field = row[0], row[1]
            symptoms = set()
            for part in symptoms_field.split(","):
                m = re.match(r"\s*(.+?)\s*\([\d.]+\)\s*", part)
                if m:
                    symptoms.add(normalize(m.group(1)))
            external[normalize(disease)] = symptoms
    return external


def load_project_edges():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        rows = session.run("""
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            RETURN d.name AS disease, s.name AS symptom
        """)
        edges = [(r["disease"], r["symptom"]) for r in rows]
    driver.close()
    return edges


def main():
    external = load_external()
    edges = load_project_edges()
    print(f"External graph: {len(external)} diseases")
    print(f"This project: {len(edges)} HAS_SYMPTOM edges")

    by_disease = {}
    for disease, symptom in edges:
        by_disease.setdefault(normalize(disease), []).append(symptom)

    overlapping_diseases = sorted(set(by_disease) & set(external))
    print(f"Overlapping diseases (exact name match): {len(overlapping_diseases)}")

    total_edges_checked = 0
    total_corroborated = 0
    per_disease = []
    for disease in overlapping_diseases:
        ext_symptoms = external[disease]
        proj_symptoms = by_disease[disease]
        corroborated = sum(1 for s in proj_symptoms if normalize(s) in ext_symptoms)
        total_edges_checked += len(proj_symptoms)
        total_corroborated += corroborated
        per_disease.append({
            "disease": disease,
            "n_project_symptoms": len(proj_symptoms),
            "n_corroborated": corroborated,
            "corroboration_rate": corroborated / len(proj_symptoms) if proj_symptoms else 0.0,
        })

    rate = total_corroborated / total_edges_checked if total_edges_checked else 0.0
    print(f"\nEdges checked (diseases with external overlap): {total_edges_checked}")
    print(f"Corroborated by external graph: {total_corroborated} ({rate*100:.1f}%)")

    uncovered = sorted(set(by_disease) - set(external))
    print(f"\nProject diseases with NO match in external graph: {len(uncovered)}")
    print(f"  (of {len(by_disease)} total diseases in this project's graph)")

    report = {
        "external_source": "Rotmensch et al. 2017 (github.com/clinicalml/HealthKnowledgeGraph)",
        "n_external_diseases": len(external),
        "n_project_diseases": len(by_disease),
        "n_overlapping_diseases": len(overlapping_diseases),
        "n_edges_checked": total_edges_checked,
        "n_corroborated": total_corroborated,
        "corroboration_rate": rate,
        "per_disease": per_disease,
        "uncovered_diseases": uncovered,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
