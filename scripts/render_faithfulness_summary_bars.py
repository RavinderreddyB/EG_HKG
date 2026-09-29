"""Bar chart of the grounded vs ungrounded faithfulness summary table
(mean faithfulness, hallucination rate) -- distinct from
render_faithfulness_distribution.py, which plots the full per-answer score
distribution. This one visualizes only the two summary numbers per
condition, matching the manuscript's faithfulness summary table exactly.

Source: logs/faithfulness_nli_final_report.json (grounded),
logs/ungrounded_ablation_report.json (ungrounded) -- avg_consistency and
avg_low_consistency_rate.

Run from the project root: python -m scripts.render_faithfulness_summary_bars
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 10

FIGDIR = "figures"

grounded = json.load(open("logs/faithfulness_nli_final_report.json"))
ungrounded = json.load(open("logs/ungrounded_ablation_report.json"))

conditions = ["Ungrounded\n(no KG)", "KG-grounded"]
n = [ungrounded["num_scored"], grounded["num_scored"]]
faithfulness = [ungrounded["avg_consistency"], grounded["avg_consistency"]]
hallucination_pct = [ungrounded["avg_low_consistency_rate"] * 100,
                      grounded["avg_low_consistency_rate"] * 100]

colors = ["#E69F00", "#0072B2"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.5))

BAR_WIDTH = 0.5
bars1 = ax1.bar(conditions, faithfulness, width=BAR_WIDTH, color=colors, edgecolor="black")
for b, v in zip(bars1, faithfulness):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.4f}",
              ha="center", fontweight="bold")
ax1.set_ylabel("Mean faithfulness (avg_consistency)")
ax1.set_ylim(0, 1.0)
ax1.set_title("Faithfulness")

bars2 = ax2.bar(conditions, hallucination_pct, width=BAR_WIDTH, color=colors, edgecolor="black")
for b, v in zip(bars2, hallucination_pct):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.2f}%",
              ha="center", fontweight="bold")
ax2.set_ylabel("Hallucination rate (avg_low_consistency_rate)")
ax2.set_ylim(0, 100)
ax2.set_title("Hallucination Rate")

fig.suptitle(f"Grounded vs Ungrounded Faithfulness Summary (n={n[0]} per arm, same disease set)",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{FIGDIR}/faithfulness_summary_bars.png", dpi=300)
plt.close()
print("Saved faithfulness_summary_bars.png")
