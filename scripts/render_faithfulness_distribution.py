"""Grounded vs ungrounded per-answer faithfulness score distribution, for
both generator families (Gemini production, Claude Sonnet 5 robustness
check) -- the full spread behind the summary means reported in Table 10.

Source: logs/faithfulness_nli_final_report.json (Gemini grounded),
logs/ungrounded_ablation_report.json (Gemini ungrounded),
logs/second_generator_claude_report.json (Claude, both arms).

Run from the project root: python -m scripts.render_faithfulness_distribution
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 10

FIGDIR = "figures"

gemini_grounded = json.load(open("logs/faithfulness_nli_final_report.json"))
gemini_ungrounded = json.load(open("logs/ungrounded_ablation_report.json"))
claude = json.load(open("logs/second_generator_claude_report.json"))

g_scores = [d["avg_consistency"] for d in gemini_grounded["per_disease"]]
u_scores = [d["avg_consistency"] for d in gemini_ungrounded["per_disease"]]
cg_scores = [d["grounded_avg_consistency"] for d in claude["per_disease"]]
cu_scores = [d["ungrounded_avg_consistency"] for d in claude["per_disease"]]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
bins = [i / 40 for i in range(0, 41)]

ax = axes[0]
u_mean, g_mean = sum(u_scores)/len(u_scores), sum(g_scores)/len(g_scores)
ax.hist(u_scores, bins=bins, alpha=0.6, color="#E69F00", edgecolor="black", linewidth=0.4, hatch="///",
        label=f"Ungrounded  mean={u_mean:.4f}")
ax.hist(g_scores, bins=bins, alpha=0.6, color="#0072B2", edgecolor="black", linewidth=0.4,
        label=f"Grounded  mean={g_mean:.4f}")
ax.axvline(u_mean, color="#E69F00", linestyle="--", linewidth=2.0)
ax.axvline(g_mean, color="#0072B2", linestyle="--", linewidth=2.0)
ax.axvline(0.5, color="#000000", linestyle=":", linewidth=1.0, label="low-consistency threshold (0.5)")
ax.set_xlabel("Per-answer faithfulness score (avg_consistency)")
ax.set_ylabel("Number of disease-answer slots")
ax.set_title(f"Gemini (n={len(g_scores)} per arm)")
ax.legend(fontsize=8)

ax = axes[1]
cu_mean, cg_mean = sum(cu_scores)/len(cu_scores), sum(cg_scores)/len(cg_scores)
ax.hist(cu_scores, bins=bins, alpha=0.6, color="#E69F00", edgecolor="black", linewidth=0.4, hatch="///",
        label=f"Ungrounded  mean={cu_mean:.4f}")
ax.hist(cg_scores, bins=bins, alpha=0.6, color="#0072B2", edgecolor="black", linewidth=0.4,
        label=f"Grounded  mean={cg_mean:.4f}")
ax.axvline(cu_mean, color="#E69F00", linestyle="--", linewidth=2.0)
ax.axvline(cg_mean, color="#0072B2", linestyle="--", linewidth=2.0)
ax.axvline(0.5, color="#000000", linestyle=":", linewidth=1.0, label="low-consistency threshold (0.5)")
ax.set_xlabel("Per-answer faithfulness score (avg_consistency)")
ax.set_title(f"Claude Sonnet 5 (n={len(cg_scores)} per arm)")
ax.legend(fontsize=8)

fig.suptitle("Grounded vs Ungrounded Faithfulness Distribution, by Generator Family", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{FIGDIR}/faithfulness_distribution.png", dpi=300)
plt.close()
print("Saved faithfulness_distribution.png")
