"""Confusion matrix for the ranking-confidence flag as a binary classifier
-- pooled over 20 seeds x 125 diseases = 2,500 held-out trials, replacing
Table 7 in the manuscript.

Source: logs/holdout_eval_multiseed_report.json -> pooled_confusion_matrix.

Run from the project root: python -m scripts.render_confusion_matrix
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm, to_rgb
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 10

FIGDIR = "figures"

report = json.load(open("logs/holdout_eval_multiseed_report.json"))
cm = report["pooled_confusion_matrix"]
n_seeds = report["n_seeds"]

# rows = predicted (flag state), cols = actual (ranking outcome) --
# matches Table 7's layout exactly
matrix = np.array([[cm["tp"], cm["fp"]],
                    [cm["fn"], cm["tn"]]])
row_labels = ["Flag raised", "No flag"]
col_labels = ["Ranking failed", "Ranking correct"]
cell_tags = np.array([["TP", "FP"], ["FN", "TN"]])

total = matrix.sum()

# Okabe-Ito colorblind-safe palette (Wong, "Points of View: Color
# Blindness," Nature Methods 8, 441, 2011) -- same blue/orange pair used
# for binary comparisons in the other figures in this project. Correct
# predictions (TP, TN) get blue; incorrect (FP, FN) get orange. A single
# sequential colormap keyed to raw count was tried first and failed: TN
# (2,216) washed out TP/FP/FN (16-192) as near-identical pale cells.
# Splitting by correctness fixes that regardless of count skew; log-scaled
# alpha within each hue still shows relative magnitude.
CORRECT_COLOR = to_rgb("#0072B2")
INCORRECT_COLOR = to_rgb("#E69F00")
cell_hue = np.array([[CORRECT_COLOR, INCORRECT_COLOR],
                      [INCORRECT_COLOR, CORRECT_COLOR]])

norm = LogNorm(vmin=matrix.min(), vmax=matrix.max())
alpha = 0.35 + 0.65 * norm(matrix)  # floor so the smallest cell isn't washed to white

rgba = np.zeros((2, 2, 4))
rgba[..., :3] = cell_hue
rgba[..., 3] = alpha

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.imshow(rgba)

ax.set_xticks([0, 1]); ax.set_xticklabels(col_labels)
ax.set_yticks([0, 1]); ax.set_yticklabels(row_labels)
ax.set_xlabel("Actual ranking outcome")
ax.set_ylabel("Predicted (flag state)")

for i in range(2):
    for j in range(2):
        v = matrix[i, j]
        pct = v / total * 100
        text_color = "white" if alpha[i, j] > 0.6 else "black"
        ax.text(j, i, f"{cell_tags[i, j]}\n{v:,}\n({pct:.1f}%)",
                ha="center", va="center", color=text_color, fontweight="bold")

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=CORRECT_COLOR, edgecolor="black", label="Correct (TP, TN)"),
    plt.Rectangle((0, 0), 1, 1, facecolor=INCORRECT_COLOR, edgecolor="black", label="Incorrect (FP, FN)"),
]
ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=2, frameon=False, fontsize=8)

ax.set_title(f"Ranking-Confidence Flag, Pooled Confusion Matrix\n"
             f"(n={total:,} trials, {n_seeds} seeds x 125 diseases)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/confusion_matrix.png", dpi=300)
plt.close()
print("Saved confusion_matrix.png")
