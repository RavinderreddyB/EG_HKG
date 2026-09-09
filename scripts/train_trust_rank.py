import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from retrieval.rag_retriever import MedicalRAG
from pipeline.trust_rank import extract_features, FEATURE_NAMES, MODEL_PATH

# scripts/run_full_evaluation.py evaluates on medquad[:50] — start training
# data after that so there's no leakage between train and eval sets.
TRAIN_QUERIES = 250
EVAL_HOLDOUT = 50


def build_training_set(rag, medquad):
    X, y = [], []

    pool = medquad[EVAL_HOLDOUT:EVAL_HOLDOUT + TRAIN_QUERIES]

    for item in tqdm(pool, desc="Building training set"):
        query = item.get("question", "")
        gold_answer = item.get("answer", "")
        if not query or not gold_answer:
            continue

        candidates = rag.get_candidates(query, disease=None, top_k=20)
        gold_norm = gold_answer.strip().lower()

        for c in candidates:
            label = 1 if c["text"].strip().lower() == gold_norm else 0
            X.append(extract_features(None, c))
            y.append(label)

    return np.array(X), np.array(y)


def main():
    with open("data/processed/medquad_filtered.json") as f:
        medquad = json.load(f)

    rag = MedicalRAG()
    X, y = build_training_set(rag, medquad)

    print(f"\nTraining examples: {len(X)} (positives: {y.sum()}, rate: {y.mean():.3f})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f} | Held-out accuracy: {test_acc:.3f}")

    print("\nLearned weights:")
    for name, coef in zip(FEATURE_NAMES, model.coef_[0]):
        print(f"  {name:16s} {coef:+.4f}")
    print(f"  {'intercept':16s} {model.intercept_[0]:+.4f}")

    from pathlib import Path
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"\n[OK] Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
