import joblib

MODEL_PATH = "models/trust_rank_model.joblib"

TREATMENT_KEYWORDS = [
    "treat", "therapy", "drug", "antibiotic",
    "medication", "prescribed", "treatment"
]

BAD_INTENT_KEYWORDS = [
    "what is", "define", "definition", "symptom", "cause"
]

FEATURE_NAMES = ["retrieval_score", "rerank_score", "disease_match", "treatment_hits", "bad_intent"]


def extract_features(disease, evidence):
    text = evidence["text"].lower()
    question = evidence.get("question", "").lower()

    disease_match = 1.0 if disease and disease.lower() in text else 0.0
    treatment_hits = sum(1 for k in TREATMENT_KEYWORDS if k in text)
    bad_intent = 1.0 if any(k in question for k in BAD_INTENT_KEYWORDS) else 0.0

    return [
        evidence.get("score", 0.0),
        evidence.get("rerank_score", 0.0),
        disease_match,
        treatment_hits,
        bad_intent,
    ]


class MedicalTrustRank:

    def __init__(self):
        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]

    def score(self, query, disease, evidence):
        features = extract_features(disease, evidence)
        return float(self.model.predict_proba([features])[0][1])

    def rank(self, query, disease, evidences):

        scored = []

        for e in evidences:
            e["trust_score"] = self.score(query, disease, e)
            scored.append(e)

        # sort highest first
        scored.sort(key=lambda x: x["trust_score"], reverse=True)

        return scored
