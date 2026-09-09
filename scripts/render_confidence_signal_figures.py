import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 10

data = json.load(open("logs/graph_stats.json"))
pagerank_scores = list(json.load(open("logs/citation_trustrank.json")).values())
struct_conf = data["structural_confidence"]

CITATION_EDGE_COUNT = 33  # live-verified via kg.citation_graph.build_citation_graph() against NCBI elink (679 nodes -> 33 edges)

FIGDIR = "figures"
import os
os.makedirs(FIGDIR, exist_ok=True)

# -----------------------------------------------------------------
# Citation-graph PageRank score distribution
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.hist(pagerank_scores, bins=30, color="#0072B2", edgecolor="black", linewidth=0.5)
baseline = 1 / len(pagerank_scores)
ax.axvline(baseline, color="#000000", linestyle="--", linewidth=1.5,
           label=f"uniform baseline (1/{len(pagerank_scores)} = {baseline:.4f})")
ax.set_xlabel("PageRank score")
ax.set_ylabel("Number of papers")
ax.set_title(f"Citation-Graph PageRank Distribution\n(n={len(pagerank_scores)} PubMed papers, {CITATION_EDGE_COUNT} internal citation edges)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/pagerank_distribution.png", dpi=300)
plt.close()

# -----------------------------------------------------------------
# TransE structural confidence distribution
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.hist(struct_conf, bins=30, color="#0072B2", edgecolor="black", linewidth=0.5)
mean_v = sum(struct_conf) / len(struct_conf)
ax.axvline(mean_v, color="#000000", linestyle="--", linewidth=1.5, label=f"mean = {mean_v:.3f}")
ax.set_xlabel("Structural confidence score (min-max normalized TransE plausibility)")
ax.set_ylabel("Number of edges")
ax.set_title(f"TransE Structural Confidence Distribution\n(n={len(struct_conf)} HAS_SYMPTOM edges)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/structural_confidence_distribution.png", dpi=300)
plt.close()

# -----------------------------------------------------------------
# Provenance coverage pie/bar
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 4.5))
labels = ["Enriched\n(real PubMed citation)", "Fallback\n(no matching paper found)"]
values = [data["enriched_edges"], data["total_edges"] - data["enriched_edges"]]
colors = ["#0072B2", "#E69F00"]
ax.bar(labels, values, color=colors, edgecolor="black")
for i, v in enumerate(values):
    pct = v / data["total_edges"] * 100
    ax.text(i, v + 5, f"{v}\n({pct:.1f}%)", ha="center", fontweight="bold")
ax.set_ylabel("Number of HAS_SYMPTOM edges")
ax.set_title(f"PubMed Provenance Coverage\n(n={data['total_edges']} total edges)")
ax.set_ylim(0, data["total_edges"] * 1.15)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/provenance_coverage.png", dpi=300)
plt.close()

print("All figures saved to", FIGDIR)
