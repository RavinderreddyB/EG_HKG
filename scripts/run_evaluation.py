import json
from pipeline.full_pipeline import MedicalPipeline
from evaluation.metrics import MedicalEvaluator

pipeline = MedicalPipeline()
evaluator = MedicalEvaluator()

# -------------------------
# Define test queries
# -------------------------
test_cases = [
    ["cough", "fever"],
    ["headache", "nausea"],
    ["chest pain", "shortness of breath"],
    ["vomiting", "diarrhea"],
    ["skin rash", "itching"],
    ["fatigue", "weight loss"],
    ["joint pain", "swelling"],
    ["abdominal pain", "yellowing of eyes"],
    ["high fever", "body pain"],
    ["breathing difficulty", "cough"]
]

results = []

for symptoms in test_cases:

    matched, unmatched = pipeline.query_engine.validate_symptoms(
        [s.strip().lower() for s in symptoms]
    )
    if not matched:
        print(f"[SKIP] {symptoms}: no recognized symptoms (unmatched: {unmatched})")
        continue

    diseases = pipeline.query_engine.find_diseases(matched)

    for d in diseases[:1]:  # only top disease

        disease_name = d["disease"]

        query = f"treatment of {disease_name} drugs therapy antibiotics"

        evidence = pipeline.rag.retrieve(query, disease=disease_name, top_k=20)
        ranked = pipeline.trust.rank(query, disease_name, evidence)

        drugs = pipeline.get_treatments(disease_name)

        try:
            answer, _ = pipeline.generator.generate(disease_name, drugs, ranked)
        except:
            answer = ""

        coverage = evaluator.evidence_coverage(answer, ranked[:3])
        hallucination = evaluator.hallucination_score(answer, ranked[:3])
        precision = evaluator.precision_at_k(ranked, k=3)
        trust_avg = evaluator.avg_trust_score(ranked, k=3)

        results.append({
            "symptoms": symptoms,
            "disease": disease_name,
            "coverage": coverage,
            "hallucination": hallucination,
            "precision": precision,
            "trust_score": trust_avg
        })

# -------------------------
# Save results
# -------------------------
with open("logs/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

pipeline.close()

print("[OK] Evaluation complete. Results saved.")