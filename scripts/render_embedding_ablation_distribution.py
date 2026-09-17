"""Figure: per-edge distribution of the embedding-model ablation -- generic
(all-MiniLM-L6-v2) vs domain-specific (PubMedBERT) semantic confidence,
across the full 770 enriched edges.

Domain-specific side uses the real stored per-edge `confidence` values
(exact, n=770). Generic side replicates kg/pubmed_linker.py's original
_compute_confidence() methodology exactly: full abstract text fetched live
via NCBI efetch, truncated to 512 characters, encoded with all-MiniLM-L6-v2.
Edges whose PMID returns no abstract (15/679 unique PMIDs) get a 0.5
fallback, matching kg/pubmed_linker.py's own fallback behavior.

Result: generic mean 0.5092, min -0.0221.

An earlier version of this ablation (data/processed/embedding_ablation_224_edges.json)
was scoped to only the 224 edges sourced from the original DiseaseAndSymptoms
dataset. The other 546 MedlinePlus-sourced enriched edges turned out to
carry real embedding-based confidence from the same pipeline (same 0.5
floor, continuous spread, pmid + relevance_score populated identically) --
there was no methodological reason to exclude them, so the ablation was
re-run over all 770 (see scripts/rerun_embedding_ablation_full.py).

Source data: data/processed/embedding_ablation_770_edges.json

Run from the project root: python -m scripts.render_embedding_ablation_distribution
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 10

DATA_PATH = "data/processed/embedding_ablation_770_edges.json"
FIGDIR = "figures"

with open(DATA_PATH) as f:
    full = json.load(f)

generic = full["generic"]
domain_specific = full["domain_specific"]

fig, ax = plt.subplots(figsize=(7, 4.8))
bins = [i / 40 for i in range(-4, 41)]
ax.hist(generic, bins=bins, alpha=0.6, color="#E69F00", edgecolor="black", linewidth=0.4, hatch="///",
        label=f"all-MiniLM-L6-v2 (generic)  mean={sum(generic)/len(generic):.3f}")
ax.hist(domain_specific, bins=bins, alpha=0.6, color="#0072B2", edgecolor="black", linewidth=0.4,
        label=f"PubMedBERT (domain-specific)  mean={sum(domain_specific)/len(domain_specific):.3f}")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Semantic confidence score (cosine similarity)")
ax.set_ylabel("Number of edges")
ax.set_title(f"Embedding Model Ablation, Per-Edge Score Distribution\n(n={len(generic)} enriched edges)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/embedding_ablation_distribution.png", dpi=300)
plt.close()
print("Saved embedding_ablation_distribution.png")
