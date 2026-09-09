"""Grounded vs ungrounded per-answer faithfulness score distribution -- the
full spread behind the 0.6426 vs 0.2107 means, n=102 per arm, same disease
set both sides.

Source: logs/faithfulness_nli_final_report.json (grounded),
logs/ungrounded_ablation_report.json (ungrounded) -- avg_consistency per
disease-answer slot.

Run from the project root: python -m scripts.render_faithfulness_distribution
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 10

FIGDIR = "figures"

grounded = json.load(open("logs/faithfulness_nli_final_report.json"))
ungrounded = json.load(open("logs/ungrounded_ablation_report.json"))

g_scores = [d["avg_consistency"] for d in grounded["per_disease"]]
u_scores = [d["avg_consistency"] for d in ungrounded["per_disease"]]

fig, ax = plt.subplots(figsize=(7, 4.8))
bins = [i / 40 for i in range(0, 41)]
ax.hist(u_scores, bins=bins, alpha=0.6, color="#E69F00", edgecolor="black", linewidth=0.4, hatch="///",
        label=f"Ungrounded  mean={sum(u_scores)/len(u_scores):.4f}")
ax.hist(g_scores, bins=bins, alpha=0.6, color="#0072B2", edgecolor="black", linewidth=0.4,
        label=f"Grounded  mean={sum(g_scores)/len(g_scores):.4f}")
ax.axvline(0.5, color="#000000", linestyle=":", linewidth=1.0, label="low-consistency threshold (0.5)")
ax.set_xlabel("Per-answer faithfulness score (avg_consistency)")
ax.set_ylabel("Number of disease-answer slots")
ax.set_title(f"Grounded vs Ungrounded Faithfulness Distribution\n(n={len(g_scores)} per arm, same disease set)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/faithfulness_distribution.png", dpi=300)
plt.close()
print("Saved faithfulness_distribution.png")
