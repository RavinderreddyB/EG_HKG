from retrieval.rag_retriever import MedicalRAG
from pipeline.trust_rank import MedicalTrustRank

rag = MedicalRAG()
trust = MedicalTrustRank()

query = "treatment of pneumonia drugs therapy antibiotics"
disease = "pneumonia"

evidence = rag.retrieve(query, disease=disease, top_k=20)

ranked = trust.rank(query, disease, evidence)

print("\nRanked Evidence:\n")

for e in ranked[:5]:
    print(f"Score: {e['trust_score']:.2f}")
    print("-", e["question"])
    print()