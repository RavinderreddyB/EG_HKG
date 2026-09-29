"""One-off export for journal submission: re-renders the manuscript
figures without any embedded title/caption text (Springer: "Do not include
titles or captions within your illustrations") and at 600 dpi TIFF
(Springer's minimum for combination artwork), into figures/submission/.

Does not touch the existing figures/*.png (those stay as-is for other uses).

Run from the project root: python -m scripts.export_publication_figures
"""
import os
import runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure

OUTDIR = "figures/submission"
os.makedirs(OUTDIR, exist_ok=True)

# Strip any embedded title/caption text -- Springer requires captions live
# only in the manuscript text, never baked into the artwork.
matplotlib.axes.Axes.set_title = lambda self, *a, **k: None
matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None

_orig_savefig = plt.savefig


def _savefig_override(fname, **kwargs):
    base = os.path.basename(str(fname))
    name = os.path.splitext(base)[0]
    new_path = os.path.join(OUTDIR, f"{name}.tiff")
    kwargs["dpi"] = 600
    kwargs["format"] = "tiff"
    print(f"  -> {new_path}")
    return _orig_savefig(new_path, **kwargs)


plt.savefig = _savefig_override

TARGET_SCRIPTS = [
    "scripts.render_confidence_signal_figures",   # pagerank, structural_confidence, provenance_coverage
    "scripts.render_signal_correlation_scatter",  # signal_correlation_scatter
    "scripts.render_embedding_ablation_distribution",  # embedding_ablation_distribution
    "scripts.render_faithfulness_distribution",   # faithfulness_distribution
    "scripts.render_confusion_matrix",            # confusion_matrix
    "scripts.render_generator_comparison_bars",   # generator_comparison_bars
]

for mod in TARGET_SCRIPTS:
    print(f"Running {mod} ...")
    try:
        runpy.run_module(mod, run_name="__main__")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nDone. Submission-ready figures (600 dpi TIFF, no embedded titles) in figures/submission/")
