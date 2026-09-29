"""Two-panel bar chart comparing the two generator families on the
grounded arm only (Section 5.7): Gemini (production fallback chain) vs
Claude Sonnet 5 (robustness check, Section 3.5/6.2).

Run from the project root: python -m scripts.render_generator_comparison_bars
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 10

FIGDIR = "figures"
import os
os.makedirs(FIGDIR, exist_ok=True)

gemini = json.load(open("logs/faithfulness_nli_final_report.json"))
claude = json.load(open("logs/second_generator_claude_report.json"))

gemini_faith = gemini["avg_consistency"]
gemini_halluc = gemini["avg_low_consistency_rate"] * 100
claude_faith = claude["grounded_mean_consistency"]
claude_halluc = claude["grounded_hallucination_rate"] * 100

models = ["Gemini\n(fallback chain)", "Claude Sonnet 5"]
colors = ["#0072B2", "#E69F00"]
BAR_WIDTH = 0.5

fig, axes = plt.subplots(1, 2, figsize=(7, 4.5))

ax = axes[0]
vals = [gemini_faith, claude_faith]
ax.bar(models, vals, width=BAR_WIDTH, color=colors, edgecolor="black")
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")
ax.set_ylabel("Mean faithfulness")
ax.set_title("Faithfulness (grounded)")
ax.set_ylim(0, 1.0)

ax = axes[1]
vals = [gemini_halluc, claude_halluc]
ax.bar(models, vals, width=BAR_WIDTH, color=colors, edgecolor="black")
for i, v in enumerate(vals):
    ax.text(i, v + 1, f"{v:.2f}%", ha="center", fontweight="bold")
ax.set_ylabel("Hallucination rate (%)")
ax.set_title("Hallucination Rate (grounded)")
ax.set_ylim(0, 100)

fig.suptitle("Grounded-Arm Faithfulness by Generator Family (n=102 per model)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{FIGDIR}/generator_comparison_bars.png", dpi=300)
plt.close()
print("Saved generator_comparison_bars.png")
