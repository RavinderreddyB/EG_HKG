"""Pairwise confidence-signal correlation scatter plots -- visualizes the
near-zero pairwise correlations (-0.03, 0.01, 0.01) between the three
confidence signals. Same min-max normalization and same 770-edge
population as compute_cci.py.

Run from the project root: python -m scripts.render_signal_correlation_scatter
"""
import matplotlib.pyplot as plt
import matplotlib
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 10

FIGDIR = "figures"


def minmax_normalize(values):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
with driver.session() as session:
    rows = session.run("""
        MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
        WHERE r.evidence_src = 'PubMed'
          AND r.confidence IS NOT NULL AND r.source_credibility IS NOT NULL AND r.structural_confidence IS NOT NULL
        RETURN r.confidence AS semantic, r.source_credibility AS credibility, r.structural_confidence AS structural
    """)
    data = [r.data() for r in rows]
driver.close()

print(f"n edges = {len(data)}")

semantic_n = minmax_normalize([r["semantic"] for r in data])
credibility_n = minmax_normalize([r["credibility"] for r in data])
structural_n = minmax_normalize([r["structural"] for r in data])

pairs = [
    ("Semantic", semantic_n, "Source Credibility", credibility_n),
    ("Semantic", semantic_n, "Structural", structural_n),
    ("Source Credibility", credibility_n, "Structural", structural_n),
]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
for ax, (xlabel, x, ylabel, y) in zip(axes, pairs):
    r = pearson(x, y)
    ax.scatter(x, y, s=10, alpha=0.4, color="#0072B2", edgecolor="none")
    ax.set_xlabel(f"{xlabel} (normalized)")
    ax.set_ylabel(f"{ylabel} (normalized)")
    ax.set_title(f"r = {r:.2f}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

fig.suptitle(f"Pairwise Confidence-Signal Correlations (n={len(data)} enriched edges)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{FIGDIR}/signal_correlation_scatter.png", dpi=300)
plt.close()
print("Saved signal_correlation_scatter.png")
