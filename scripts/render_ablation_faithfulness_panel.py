"""Composite figure combining the two two-distribution comparison histograms
into one side-by-side panel, to stay under the journal's combined
figure+table limit:
  (A) embedding_ablation_distribution.png -- generic vs domain-specific
      semantic confidence, per-edge
  (B) faithfulness_distribution.png       -- grounded vs ungrounded
      per-answer faithfulness

Composites the already-rendered PNGs rather than re-deriving from source
data, matching render_confidence_panel.py's approach.

Run from the project root, after both source figures already exist:
    python -m scripts.render_ablation_faithfulness_panel
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"

FIGDIR = "figures"
PANELS = [
    ("A", f"{FIGDIR}/embedding_ablation_distribution.png"),
    ("B", f"{FIGDIR}/faithfulness_distribution.png"),
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

plt.savefig(f"{FIGDIR}/ablation_faithfulness_panel.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved ablation_faithfulness_panel.png")
