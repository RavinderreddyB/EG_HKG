import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Biomedical sentence embedding model, used by kg/pubmed_linker.py,
# retrieval/rag_retriever.py, and scripts/build_vector_index.py.
# Swapped from all-MiniLM-L6-v2 (general-purpose) for domain-specific
# accuracy on PubMed/medical text. Changing this requires rebuilding
# the FAISS index and re-running provenance enrichment.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")

# Cross-encoder reranker, used by retrieval/rag_retriever.py to rescore the
# FAISS candidate pool by true (query, answer) relevance instead of a
# bag-of-treatment-keywords heuristic.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")