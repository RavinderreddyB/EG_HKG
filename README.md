# EG-HKG: Evidence-Grounded Healthcare Knowledge Graph

A knowledge-graph-enhanced language model system for symptom-driven clinical
question answering. Every disease-symptom relationship in the graph is
enriched, where possible, with a real PubMed citation and scored along three
independent confidence signals — semantic relevance, source credibility, and
structural plausibility — rather than a single opaque confidence number.

Given a set of symptoms, the system ranks candidate diseases from a Neo4j
knowledge graph, retrieves supporting evidence (PubMed citations for the
original disease set, direct MedlinePlus extracts for the expanded set), and
generates a grounded answer covering the condition, its treatments, and any
drug-drug interaction warnings — using an LLM constrained to only the
evidence it was given.

## What's in the graph

- 125 diseases, 439 symptoms (Kaggle disease-symptom dataset + MedlinePlus)
- 928 disease-symptom edges, 83% with a real retrievable PubMed citation
- 105 drug-treatment edges, 241 non-pharmacological treatment edges
- 191,252 drug-drug interaction edges, 3,543 symptom co-occurrence edges

## Architecture

```
Data sources (Kaggle CSV, MedlinePlus XML, SIDER/MedDRA, drug interactions)
        │
        ▼
Neo4j Knowledge Graph  (kg/builder.py, kg/drug_builder.py, kg/interaction_builder.py)
        │
        ▼
PubMed Provenance Enrichment  (kg/pubmed_linker.py, kg/citation_graph.py)
   → semantic confidence, source-credibility (PageRank), structural
     confidence (TransE) computed per edge
        │
        ▼
Query Engine  (pipeline/query_engine.py)
   → symptom validation, IDF+co-occurrence disease ranking,
     ranking-confidence flag (check_ranking_confidence)
        │
        ├─→ Original disease set: MedQuAD dense retrieval + cross-encoder
        │   reranking + evidence-trust classifier (retrieval/rag_retriever.py,
        │   pipeline/trust_rank.py)
        │
        └─→ MedlinePlus disease set: direct extracted evidence
            (bypasses retrieval entirely)
        │
        ▼
Generation  (pipeline/generator.py, Gemini with model fallback chain)
   → grounded answer: explanation, treatment, drug-interaction warnings
        │
        ▼
Faithfulness Evaluation  (evaluation/faithfulness_nli.py)
   → HHEM-based per-sentence entailment scoring against the supplied evidence
```

## Project structure

```
EG_HKG/
├── config/          # Neo4j credentials, embedding/reranker model names (config/settings.py)
├── kg/              # graph construction, PubMed enrichment, citation-graph PageRank
├── pipeline/        # the orchestrator (full_pipeline.py), ranking, trust classifier, generation
├── retrieval/       # FAISS dense retrieval + cross-encoder reranking
├── evaluation/       # HHEM faithfulness scorer + dev-only sanity-check metrics
├── utils/           # PubChem CID → drug name lookup helper
├── scripts/         # build pipeline, evaluation, and figure-rendering scripts (see below)
├── data/            # small committed datasets; large/third-party data is gitignored, see data/README.md
├── models/          # trained evidence-trust classifier (trust_rank_model.joblib)
├── baselines/       # frozen pre-PubMedBERT snapshot, kept for the embedding-model ablation
├── logs/            # evaluation reports (each one backs a specific result)
└── Figures/         # generated result figures, one script per figure in scripts/
```

## Setup

1. **Clone and create a virtual environment:**
   ```bash
   git clone <this-repo>
   cd EG_HKG
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   pip install -r requirements.txt
   ```

2. **Configure environment variables** — copy `.env.example` to `.env` and fill in:
   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — your Neo4j instance
   - `GEMINI_API_KEY` — free at https://aistudio.google.com
   - `NCBI_API_KEY` — optional, free at https://www.ncbi.nlm.nih.gov/account; raises the PubMed enrichment rate limit from 3 to 10 requests/second

3. **Start Neo4j** — either Neo4j Desktop, or `docker-compose up -d` (the compose file maps to the standard `bolt://localhost:7687`, matching `.env.example`).

## Building the graph and data from scratch

Some large/third-party inputs are not committed to the repo (see
`data/README.md` for the full list and reasoning). Rebuild them first:

```bash
python -m scripts.download_medquad          # -> data/raw/medquad/medquad.jsonl
python -m scripts.process_medquad            # -> data/processed/medquad.json
python -m scripts.filter_medquad             # -> data/processed/medquad_filtered.json
python -m scripts.build_vector_index         # -> data/vector_store/medquad.faiss
```

**Note:** `build_vector_index.py` encodes ~47,000 entries with a domain-specific
transformer. On CPU (no GPU) this takes roughly 3 hours — budget for it, it's
not stuck.

The MedlinePlus raw XML has no fetch script (NLM doesn't version the feed) —
download manually from https://medlineplus.gov/xml.html if you need to
re-run `extract_medlineplus_evidence.py`.

## Building the knowledge graph

```bash
python -m scripts.build_kg                  # disease-symptom graph from the Kaggle CSV
python -m scripts.build_drug_kg              # drug-disease edges
python -m scripts.build_interactions         # drug-drug interaction edges
python -m scripts.add_provenance             # PubMed citation enrichment (slow, crash-safe/resumable)
python -m scripts.build_citation_trustrank   # source-credibility signal (PageRank)
python -m scripts.build_kg_embeddings        # structural-confidence signal (TransE)
python -m scripts.train_trust_rank           # evidence-trust classifier
```

## Running the system

```bash
python -m scripts.test_pipeline
```

Prompts for a comma-separated symptom list and runs the full pipeline —
ranking, evidence retrieval, generation, and interaction checking — printing
the result and saving a run record to `logs/pipeline_runs/`.

## Reproducing the evaluation results

```bash
python -m scripts.compute_holdout_eval       # 20-seed held-out ranking evaluation
python -m scripts.generate_batch_runs 34 777 # the query batch faithfulness is scored on
python -m scripts.eval_faithfulness_final    # grounded faithfulness/hallucination scoring
python -m scripts.eval_ungrounded_ablation   # no-KG baseline, for comparison
python -m scripts.compute_arda               # ranking-confidence flag evaluation
python -m scripts.compute_cci                # confidence-signal independence check
python -m scripts.gather_graph_stats         # -> logs/graph_stats.json
python -m scripts.compute_all_metrics        # consolidates everything above
```

Figures in `figures/` are each produced by a dedicated script in
`scripts/` (`render_confidence_signal_figures.py`,
`render_signal_correlation_scatter.py`,
`render_embedding_ablation_distribution.py`,
`render_faithfulness_distribution.py`, `render_kg_graph.py`).

## Known limitations

- Source-credibility (PageRank) and structural-confidence (TransE) signals
  are both weak at the current graph scale — reported in full rather than
  omitted; see the evaluation logs for exact numbers.
- Evidence-trust classification (`pipeline/trust_rank.py`) only runs on the
  original disease set's retrieval path; the MedlinePlus-sourced set uses a
  fixed trust value instead.
- All ranking evaluation is internal (the graph tested against its own
  associations) — it does not establish clinical correctness against an
  external ground truth.
