"""Regenerate the evidence-assembly half of the pipeline (drugs, treatment
modalities, PubMed citations, trusted evidence) for a fresh batch of
symptom queries -- WITHOUT calling the Gemini generator. The original
run_*.json files used for the grounded/ungrounded faithfulness evaluation
no longer exist on disk, so this rebuilds the same kind of evidence-only
records needed as input for a second-generator (Claude) comparison,
mirroring pipeline/full_pipeline.py's run() lines 240-329 exactly but
skipping the generation step entirely -- zero Gemini calls, zero Gemini
quota spent.

n=34, seed=777 matches the methodology already reported for the grounded/
ungrounded faithfulness evaluation (Section 5.8 / 6.2).

Run from the project root: python -m scripts.generate_batch_evidence_only
"""
import json
import random
import time
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

from pipeline.query_engine import SymptomQueryEngine
from retrieval.rag_retriever import MedicalRAG
from pipeline.trust_rank import MedicalTrustRank
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

N_QUERIES = 34
SEED = 777
OUTPUT_DIR = Path("logs/evidence_only_runs")


class EvidenceAssembler:
    def __init__(self):
        self.query_engine = SymptomQueryEngine()
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.rag = MedicalRAG()
        self.trust = MedicalTrustRank()

        self.mapping = pd.read_csv("data/raw/drugs/cid_to_name.csv")
        self.cid_to_name = dict(zip(self.mapping["cid"].astype(str), self.mapping["name"]))

        medlineplus_path = Path("data/vector_store/medlineplus_evidence.json")
        with open(medlineplus_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.medlineplus_evidence = {name.lower(): entry for name, entry in raw.items()}

    def close(self):
        self.query_engine.close()
        self.driver.close()

    def get_treatments(self, disease):
        query = """
        MATCH (d:Disease {name: $disease})<-[:TREATS]-(dr:Drug)
        RETURN dr.name AS drug LIMIT 20
        """
        with self.driver.session() as session:
            results = session.run(query, disease=disease)
            drugs = []
            for r in results:
                name = self.cid_to_name.get(r["drug"])
                if name:
                    drugs.append(name)
            return drugs[:5]

    def get_treatment_modalities(self, disease):
        query = """
        MATCH (d:Disease {name: $disease})-[:HAS_TREATMENT]->(t:Treatment)
        RETURN t.name AS treatment LIMIT 10
        """
        with self.driver.session() as session:
            results = session.run(query, disease=disease)
            return [r["treatment"] for r in results]

    def get_medlineplus_evidence(self, disease_name):
        entry = self.medlineplus_evidence.get(disease_name.lower())
        if not entry:
            return None
        sections = [s.strip() for s in entry["text"].split("\n\n") if s.strip()]
        sections.sort(key=lambda s: 0 if "treat" in s.lower()[:120] else 1)
        return [
            {"question": disease_name, "text": s, "trust_score": 1.0, "source_url": entry.get("url")}
            for s in sections
        ]

    def get_citations(self, disease):
        query = """
        MATCH (d:Disease {name: $disease})-[r:HAS_SYMPTOM]->(s:Symptom)
        WHERE r.evidence_src = 'PubMed'
        RETURN s.name AS symptom, r.pmid AS pmid, r.pub_title AS title,
               r.pub_year AS year, r.confidence AS confidence,
               r.source_credibility AS source_credibility,
               r.structural_confidence AS structural_confidence,
               r.abstract_snippet AS abstract_snippet
        ORDER BY r.confidence DESC
        LIMIT 5
        """
        with self.driver.session() as session:
            results = session.run(query, disease=disease)
            return [r.data() for r in results]

    def assemble(self, symptoms):
        matched, unmatched = self.query_engine.validate_symptoms(symptoms)
        if not matched:
            return None

        diseases = self.query_engine.find_diseases(matched)
        resolved_symptoms = sorted({node for nodes in matched.values() for node in nodes})
        run_record = {"symptoms": symptoms, "resolved_symptoms": resolved_symptoms, "diseases": []}

        for d in diseases[:3]:
            disease_name = d["disease"]
            disease_record = {"disease": disease_name}

            drugs = self.get_treatments(disease_name)
            disease_record["drugs"] = drugs

            treatment_modalities = [] if drugs else self.get_treatment_modalities(disease_name)
            disease_record["treatment_modalities"] = treatment_modalities

            citations = self.get_citations(disease_name)
            disease_record["citations"] = citations

            medlineplus_sections = self.get_medlineplus_evidence(disease_name)
            if medlineplus_sections is not None:
                ranked = medlineplus_sections
                disease_record["evidence_source"] = "MedlinePlus_direct"
            else:
                section_queries = [
                    f"what is {disease_name}",
                    f"treatment of {disease_name} drugs therapy antibiotics",
                    f"{disease_name} safety precautions warnings",
                ]
                combined = []
                for q in section_queries:
                    candidates = self.rag.retrieve(q, disease=disease_name, top_k=20)
                    ranked_q = self.trust.rank(q, disease_name, candidates)
                    combined.extend(ranked_q[:2])
                seen_text = set()
                ranked = []
                for e in combined:
                    if e["text"] not in seen_text:
                        seen_text.add(e["text"])
                        ranked.append(e)
                ranked.sort(key=lambda e: e["trust_score"], reverse=True)
                disease_record["evidence_source"] = "MedQuAD_RAG"

            disease_record["ranked_evidence"] = ranked[:8] if ranked else []
            disease_record["trusted_evidence"] = [
                {"question": e["question"], "text": e["text"], "trust_score": e["trust_score"]}
                for e in ranked[:8]
            ] if ranked else []

            run_record["diseases"].append(disease_record)

        return run_record


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        symptoms_pool = [r["name"] for r in session.run("MATCH (s:Symptom) RETURN s.name AS name")]
    driver.close()
    print(f"Symptom pool size: {len(symptoms_pool)}")

    random.seed(SEED)
    all_combos = [random.sample(symptoms_pool, 3) for _ in range(N_QUERIES)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assembler = EvidenceAssembler()

    total_disease_records = 0
    for i, symptoms in enumerate(all_combos):
        print(f"\n===== [{i + 1}/{N_QUERIES}] symptoms={symptoms} =====", flush=True)
        try:
            record = assembler.assemble(symptoms)
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            continue
        if not record:
            print("[skip] no matched symptoms", flush=True)
            continue

        run_id = f"{i:03d}_{time.strftime('%H%M%S')}"
        with open(OUTPUT_DIR / f"evidence_{run_id}.json", "w") as f:
            json.dump(record, f, indent=2)
        total_disease_records += len(record["diseases"])
        print(f"  -> {len(record['diseases'])} disease records saved", flush=True)

    assembler.close()
    print(f"\n[OK] Done. Total disease-answer slots assembled: {total_disease_records}")


if __name__ == "__main__":
    main()
