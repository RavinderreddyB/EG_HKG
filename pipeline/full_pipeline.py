from pipeline.query_engine import SymptomQueryEngine
from retrieval.rag_retriever import MedicalRAG
from pipeline.trust_rank import MedicalTrustRank
from pipeline.generator import MedicalGenerator

from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import pandas as pd
import os
import time
import json
from pathlib import Path

RUNS_DIR = Path("logs/pipeline_runs")


class MedicalPipeline:

    def __init__(self):

        # -------------------------
        # KG Query Engine
        # -------------------------
        self.query_engine = SymptomQueryEngine()

        # -------------------------
        # Neo4j Connection
        # -------------------------
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        # -------------------------
        # RAG + TrustRank
        # -------------------------
        self.rag = MedicalRAG()
        self.trust = MedicalTrustRank()

        # -------------------------
        # Generator (Gemini)
        # -------------------------
        self.generator = MedicalGenerator(api_key=os.getenv("GEMINI_API_KEY"))

        # -------------------------
        # Drug Mapping (CID -> Name)
        # -------------------------
        self.mapping = pd.read_csv("data/raw/drugs/cid_to_name.csv")
        self.cid_to_name = dict(zip(self.mapping["cid"].astype(str), self.mapping["name"]))

        # -------------------------
        # MedlinePlus direct evidence (bypasses MedQuAD RAG for the 84
        # MedlinePlus-sourced diseases, which MedQuAD has near-zero real
        # coverage for -- confirmed via direct check on "Dislocated Shoulder":
        # MedQuAD's top retrieved passages were unrelated rare-genetic-
        # disorder entries matched on incidental keyword overlap, not the
        # disease itself)
        # -------------------------
        medlineplus_path = Path("data/vector_store/medlineplus_evidence.json")
        with open(medlineplus_path, encoding="utf-8") as f:
            raw_medlineplus_evidence = json.load(f)
        # find_diseases() returns disease names lowercased; index by lowercase
        # so the lookup in get_medlineplus_evidence matches regardless of case
        self.medlineplus_evidence = {
            name.lower(): entry for name, entry in raw_medlineplus_evidence.items()
        }

    def close(self):
        self.query_engine.close()
        self.driver.close()

    # -------------------------
    # Get Treatments from KG
    # -------------------------
    def get_treatments(self, disease):

        query = """
        MATCH (d:Disease {name: $disease})<-[:TREATS]-(dr:Drug)
        RETURN dr.name AS drug LIMIT 20
        """

        with self.driver.session() as session:
            results = session.run(query, disease=disease)

            drugs = []

            for r in results:
                cid = r["drug"]
                name = self.cid_to_name.get(cid)

                if name:
                    drugs.append(name)

            return drugs[:5]

    # -------------------------
    # Get Treatment Modalities from KG (Tier 2 fallback —
    # named approaches like "chemotherapy", not specific drugs)
    # -------------------------
    def get_treatment_modalities(self, disease):

        query = """
        MATCH (d:Disease {name: $disease})-[:HAS_TREATMENT]->(t:Treatment)
        RETURN t.name AS treatment LIMIT 10
        """

        with self.driver.session() as session:
            results = session.run(query, disease=disease)
            return [r["treatment"] for r in results]

    # -------------------------
    # Get MedlinePlus Direct Evidence (real source text, not MedQuAD RAG)
    # -------------------------
    def get_medlineplus_evidence(self, disease_name):
        entry = self.medlineplus_evidence.get(disease_name.lower())
        if not entry:
            return None

        # split the topic's full-summary into its heading-delimited sections
        # so faithfulness scoring can match sentences against the specific
        # section they draw from, instead of one large blob that gets cut
        # off by the evaluator's per-piece character cap
        sections = [s.strip() for s in entry["text"].split("\n\n") if s.strip()]

        # surface the treatment section first so it always survives any
        # downstream cap on how many evidence pieces get used
        sections.sort(key=lambda s: 0 if "treat" in s.lower()[:120] else 1)

        return [
            {"question": disease_name, "text": s, "trust_score": 1.0, "source_url": entry.get("url")}
            for s in sections
        ]

    # -------------------------
    # Get Symptom Citations from KG (Provenance)
    # -------------------------
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

    # -------------------------
    # Drug Interaction Check
    # -------------------------
    def check_interactions(self, drugs):

        if not drugs:
            return []

        query = """
        MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
        WHERE d1.name IN $drugs AND d2.name IN $drugs
        RETURN d1.name, d2.name, r.description
        """

        with self.driver.session() as session:
            results = session.run(query, drugs=drugs)
            return [r.data() for r in results]

    # -------------------------
    # MAIN PIPELINE
    # -------------------------
    def run(self, symptoms):

        print("\nInput Symptoms:", symptoms)

        # STEP 0 — Validate Symptoms Against KG
        matched, unmatched = self.query_engine.validate_symptoms(symptoms)

        if unmatched:
            print(f"\n[WARN] Unrecognized symptoms (ignored): {unmatched}")

        if not matched:
            print("\n[ERROR] None of the entered symptoms exist in the knowledge graph. Try different terms.")
            return

        print(f"[OK] Using recognized symptoms: {list(matched.keys())}")

        # STEP 1 — KG Disease Retrieval
        diseases = self.query_engine.find_diseases(matched)

        # STEP 1.5 — Ranking Confidence Check: warn (not block) when several
        # diseases tie on match_count with the top result, since the ranking
        # formula has no real basis to prefer one over another in that case
        # -- same caution a clinician would apply to a report of only
        # common, non-specific symptoms rather than committing to one
        # diagnosis. Checked against the actual ranked output, not guessed
        # from input symptom frequency (see check_ranking_confidence).
        confidence = self.query_engine.check_ranking_confidence(diseases)
        if confidence["is_low_confidence"]:
            print(
                f"\n[NOTE] Low-confidence ranking: {len(confidence['tied_diseases'])} diseases "
                f"tie on {confidence['top_match_count']} matched symptoms with no clear leader "
                f"-- {', '.join(confidence['tied_diseases'][:6])}"
                f"{'...' if len(confidence['tied_diseases']) > 6 else ''}. "
                "Results below may be less reliable; consider adding more distinctive "
                "symptoms for a more confident match."
            )

        print("\nTop Diseases:")
        for d in diseases[:3]:
            print(d)

        all_drugs = []
        resolved_symptoms = sorted({node for nodes in matched.values() for node in nodes})
        run_record = {
            "symptoms": symptoms,
            "resolved_symptoms": resolved_symptoms,
            "low_confidence_ranking": confidence["is_low_confidence"],
            "tied_diseases": confidence["tied_diseases"] if confidence["is_low_confidence"] else [],
            "diseases": [],
        }

        print("\nTreatments + Evidence + Final Answer:")

        # STEP 2 — Process Each Disease
        for d in diseases[:3]:

            disease_name = d["disease"]
            disease_record = {"disease": disease_name}

            print("\n==============================")
            print(f"Disease: {disease_name}")
            print("==============================")

            # -------------------------
            # KG Treatments
            # -------------------------
            drugs = self.get_treatments(disease_name)

            print("\nDrugs:")
            print(drugs if drugs else "No mapped drugs found")

            all_drugs.extend(drugs)
            disease_record["drugs"] = drugs

            # Tier 2 fallback: no specific drug found -> check treatment
            # modalities (e.g. "chemotherapy", "psychotherapy") before
            # falling through to the "no data" case in the generator.
            treatment_modalities = [] if drugs else self.get_treatment_modalities(disease_name)
            if treatment_modalities:
                print("\nTreatment Modalities (no specific drug in KG):")
                print(treatment_modalities)
            disease_record["treatment_modalities"] = treatment_modalities

            # -------------------------
            # KG Citations (Provenance)
            # -------------------------
            citations = self.get_citations(disease_name)

            print("\nEvidence Citations (Confidence):")
            if citations:
                for c in citations:
                    cred = c.get("source_credibility")
                    cred_str = f" | source credibility {cred:.4f}" if cred is not None else ""
                    struct = c.get("structural_confidence")
                    struct_str = f" | structural confidence {struct:.4f}" if struct is not None else ""
                    print(f"- {c['symptom']}: PMID {c['pmid']} ({c['year']}) | confidence {c['confidence']:.2f}{cred_str}{struct_str} | {c['title'][:80]}")
            else:
                print("No PubMed-backed citations found for this disease")

            disease_record["citations"] = citations

            # -------------------------
            # RAG Retrieval -- one query per answer section instead of a
            # single generic query, so retrieval doesn't have to serve
            # "explain the disease", "list treatment", and "note safety"
            # all at once (a single treatment-focused query was pulling
            # off-topic evidence for sections it wasn't built for, e.g.
            # bronchial asthma retrieving Bronchitis/Bronchiectasis passages)
            # -------------------------
            medlineplus_sections = self.get_medlineplus_evidence(disease_name)

            if medlineplus_sections is not None:
                ranked = medlineplus_sections
                disease_record["evidence_source"] = "MedlinePlus_direct"

                print("\nTrusted Evidence (MedlinePlus direct source, not MedQuAD RAG):")
                for e in ranked[:3]:
                    print(f"- {e['text'][:80]}...")
            else:
                section_queries = [
                    f"what is {disease_name}",
                    f"treatment of {disease_name} drugs therapy antibiotics",
                    f"{disease_name} safety precautions warnings",
                ]
                query = section_queries[1]  # kept for TrustRank's disease_match/bad_intent features and citations lookup

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

                print("\nTrusted Evidence:")

                if ranked:
                    for e in ranked[:3]:
                        print(f"- {e['question']} (Score: {e['trust_score']:.2f})")
                else:
                    print("No reliable evidence found")

            disease_record["trusted_evidence"] = [
                {"question": e["question"], "text": e["text"], "trust_score": e["trust_score"]}
                for e in ranked[:8]
            ] if ranked else []

            # -------------------------
            # FINAL GENERATION (KEY STEP)
            # -------------------------
            print("\nFinal Answer:\n")

            answer = None
            generation_model = None
            for attempt in range(3):
                try:
                    answer, generation_model = self.generator.generate(disease_name, drugs, ranked, citations, treatment_modalities=treatment_modalities)
                    print(answer)
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"[WARN] Retrying ({attempt + 1}/2)... {str(e)[:80]}")
                        time.sleep(3)
                    else:
                        print("[WARN] Generation failed after 3 attempts:", str(e))

            disease_record["answer"] = answer
            disease_record["generation_model"] = generation_model
            run_record["diseases"].append(disease_record)

        # -------------------------
        # STEP 3 — Interaction Check
        # -------------------------
        print("\nChecking interactions:")

        interactions = self.check_interactions(all_drugs)

        if interactions:
            for i in interactions:
                print(i)
        else:
            print("No interactions found")

        run_record["interactions"] = interactions

        # -------------------------
        # STEP 4 — Export Results
        # -------------------------
        self._export_run(run_record)

    # -------------------------
    # Export run to JSON + Markdown
    # -------------------------
    def _export_run(self, run_record):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime("%Y%m%d_%H%M%S")

        json_path = RUNS_DIR / f"run_{run_id}.json"
        with open(json_path, "w") as f:
            json.dump(run_record, f, indent=2)

        md_path = RUNS_DIR / f"run_{run_id}.md"
        lines = [f"# Pipeline Run — {run_id}", f"\n**Symptoms:** {', '.join(run_record['symptoms'])}\n"]

        if run_record.get("low_confidence_ranking"):
            lines.append(
                f"> **Note:** Low-confidence ranking — {len(run_record['tied_diseases'])} diseases "
                f"tied on matched-symptom count with no clear leader: {', '.join(run_record['tied_diseases'][:6])}"
                f"{'...' if len(run_record['tied_diseases']) > 6 else ''}. Results below may be less "
                "reliable; consider adding more distinctive symptoms for a more confident match.\n"
            )

        for d in run_record["diseases"]:
            lines.append(f"## {d['disease']}")
            lines.append(f"**Drugs:** {', '.join(d['drugs']) if d['drugs'] else 'None mapped'}\n")

            if d["citations"]:
                lines.append("**Evidence Citations:**")
                for c in d["citations"]:
                    cred = c.get("source_credibility")
                    cred_str = f", source credibility {cred:.4f}" if cred is not None else ""
                    struct = c.get("structural_confidence")
                    struct_str = f", structural confidence {struct:.4f}" if struct is not None else ""
                    lines.append(f"- {c['symptom']}: [PMID {c['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{c['pmid']}) ({c['year']}), confidence {c['confidence']:.2f}{cred_str}{struct_str}")
                lines.append("")

            if d["answer"]:
                lines.append(f"**Answer** (generated by `{d.get('generation_model')}`):")
                lines.append(d["answer"])
                lines.append("")

        if run_record["interactions"]:
            lines.append("## Drug Interactions")
            for i in run_record["interactions"]:
                lines.append(f"- {i}")

        with open(md_path, "w") as f:
            f.write("\n".join(lines))

        print(f"\nRun exported to {json_path} and {md_path}")