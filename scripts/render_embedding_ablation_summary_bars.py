"""Bar chart of the embedding-model ablation summary table (mean semantic
confidence, minimum score per model) -- distinct from
render_embedding_ablation_distribution.py, which plots the full per-edge
score distribution. This one visualizes only the two summary numbers per
model, matching the manuscript's Table 9 exactly.

Source: data/processed/embedding_ablation_770_edges.json (full enriched
population -- see scripts/rerun_embedding_ablation_full.py for why the
earlier 224-edge scope was widened).

Run from the project root: python -m scripts.render_embedding_ablation_summary_bars
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 10

FIGDIR = "figures"
DATA_PATH = "data/processed/embedding_ablation_770_edges.json"

full = json.load(open(DATA_PATH))
generic = full["generic"]
domain_specific = full["domain_specific"]

models = ["all-MiniLM-L6-v2\n(generic)", "PubMedBERT\n(domain-specific)"]
means = [sum(generic) / len(generic), sum(domain_specific) / len(domain_specific)]
mins = [min(generic), min(domain_specific)]

colors = ["#E69F00", "#0072B2"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

bars1 = ax1.bar(models, means, color=colors, edgecolor="black")
for b, v in zip(bars1, means):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
              ha="center", fontweight="bold")
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_ylabel("Mean semantic confidence")
ax1.set_ylim(0, 1.0)
ax1.set_title("Mean Score")

bars2 = ax2.bar(models, mins, color=colors, edgecolor="black")
for b, v in zip(bars2, mins):
    offset = 0.02 if v >= 0 else -0.045
    ax2.text(b.get_x() + b.get_width() / 2, v + offset, f"{v:.3f}",
              ha="center", fontweight="bold")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("Minimum semantic confidence")
ax2.set_ylim(-0.1, 0.6)
ax2.set_title("Minimum Score")

fig.suptitle(f"Embedding Model Ablation Summary (n={len(generic)} PubMed-cited edges)",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{FIGDIR}/embedding_ablation_summary_bars.png", dpi=300)
plt.close()
print("Saved embedding_ablation_summary_bars.png")
