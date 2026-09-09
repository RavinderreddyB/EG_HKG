from retrieval.rag_retriever import MedicalRAG

rag = MedicalRAG()

query = "treatment of pneumonia drugs therapy antibiotics"

results = rag.retrieve(query, disease="pneumonia", top_k=20)

print("\nQuery:", query)
print("\nRetrieved Evidence:\n")

for r in results:
    print("-", r["question"])