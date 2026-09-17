"""Composite figure combining the three confidence-signal diagnostics into
one multi-panel image, to stay under the journal's combined figure+table
limit:
  (A) pagerank_distribution.png       -- source-credibility signal
  (B) structural_confidence_distribution.png -- structural-confidence signal
  (C) confusion_matrix.png            -- ranking-confidence flag, pooled
  (D) signal_correlation_scatter.png  -- pairwise correlation between all three

Composites the already-rendered PNGs rather than re-deriving from source
data, since (D) requires a live Neo4j connection and the others don't --
keeping this script dependency-free and fast to re-run after any single
panel changes.

Run from the project root, after the four source figures already exist:
    python -m scripts.render_confidence_panel
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"

FIGDIR = "figures"
PANELS_TOP = [
    ("A", f"{FIGDIR}/pagerank_distribution.png"),
    ("B", f"{FIGDIR}/structural_confidence_distribution.png"),
    ("C", f"{FIGDIR}/confusion_matrix.png"),
]
PANEL_BOTTOM = ("D", f"{FIGDIR}/signal_correlation_scatter.png")

imgs_top = [plt.imread(path) for _, path in PANELS_TOP]
img_bottom = plt.imread(PANEL_BOTTOM[1])

# Row 1: same target height, each column's width follows its own image's
# aspect ratio (they're not the same shape -- confusion_matrix carries a
# legend and is taller/squarer than the two histograms).
ROW1_HEIGHT_IN = 4.2
col_widths = [ROW1_HEIGHT_IN * (im.shape[1] / im.shape[0]) for im in imgs_top]
fig_width = sum(col_widths)

# Row 2: single wide panel spanning the full figure width, height set by
# its own aspect ratio so it isn't stretched or squashed.
row2_aspect = img_bottom.shape[1] / img_bottom.shape[0]
row2_height = fig_width / row2_aspect

fig_height = ROW1_HEIGHT_IN + row2_height
fig = plt.figure(figsize=(fig_width, fig_height))

x = 0.0
for (label, _), img, w in zip(PANELS_TOP, imgs_top, col_widths):
    ax = fig.add_axes([x / fig_width, row2_height / fig_height,
                        w / fig_width, ROW1_HEIGHT_IN / fig_height])
    ax.imshow(img)
    ax.axis("off")
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=14, fontweight="bold",
             ha="left", va="top")
    x += w

ax_bottom = fig.add_axes([0, 0, 1, row2_height / fig_height])
ax_bottom.imshow(img_bottom)
ax_bottom.axis("off")
ax_bottom.text(0.01, 0.97, PANEL_BOTTOM[0], transform=ax_bottom.transAxes, fontsize=14,
               fontweight="bold", ha="left", va="top")

plt.savefig(f"{FIGDIR}/confidence_signals_panel.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved confidence_signals_panel.png")
