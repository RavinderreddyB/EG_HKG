"""Composite figure combining the embedding-model ablation distribution
with the grounded/ungrounded faithfulness *summary* bars (not the full
distribution histogram -- see render_ablation_faithfulness_panel.py for
that pairing):
  (A) embedding_ablation_distribution.png -- generic vs domain-specific
      semantic confidence, per-edge, full 770-edge population
  (B) faithfulness_summary_bars.png       -- mean faithfulness and
      hallucination rate, grounded vs ungrounded

Composites the already-rendered PNGs rather than re-deriving from source
data, matching render_confidence_panel.py's approach.

Run from the project root, after both source figures already exist:
    python -m scripts.render_ablation_faithfulness_summary_panel
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"

FIGDIR = "figures"
PANELS = [
    ("A", f"{FIGDIR}/embedding_ablation_distribution.png"),
    ("B", f"{FIGDIR}/faithfulness_summary_bars.png"),
]

imgs = [plt.imread(path) for _, path in PANELS]

ROW_HEIGHT_IN = 4.5
col_widths = [ROW_HEIGHT_IN * (im.shape[1] / im.shape[0]) for im in imgs]
fig_width = sum(col_widths)

fig = plt.figure(figsize=(fig_width, ROW_HEIGHT_IN))

x = 0.0
for (label, _), img, w in zip(PANELS, imgs, col_widths):
    ax = fig.add_axes([x / fig_width, 0, w / fig_width, 1])
    ax.imshow(img)
    ax.axis("off")
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=14, fontweight="bold",
             ha="left", va="top")
    x += w

plt.savefig(f"{FIGDIR}/ablation_faithfulness_summary_panel.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved ablation_faithfulness_summary_panel.png")
