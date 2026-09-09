"""Render a disease/symptom/drug/interaction graph PNG for a pipeline run.

Usage:
    python -m scripts.render_kg_graph                  # latest run in logs/pipeline_runs/
    python -m scripts.render_kg_graph logs/pipeline_runs/run_20260724_164217.json

Drugs and interactions are read directly from the run's JSON export (exactly
what the pipeline showed the user); only the disease->symptom edges come from
a live KG query, since the run export doesn't store that mapping per-disease.
"""
import sys
import json
import textwrap
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

matplotlib.rcParams["font.family"] = "Times New Roman"

RUNS_DIR = Path("logs/pipeline_runs")

SYMPTOM_QUERY = """
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
WHERE d.name IN $diseases AND s.name IN $symptoms
RETURN d.name AS disease, s.name AS symptom
"""


def latest_run_path():
    runs = sorted(RUNS_DIR.glob("run_*.json"))
    if not runs:
        raise SystemExit(f"No run files found in {RUNS_DIR}")
    return runs[-1]


def main():
    run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run_path()
    run = json.loads(run_path.read_text())

    diseases = [d["disease"] for d in run.get("diseases", [])]
    # resolved_symptoms holds the actual KG node names (e.g. "high fever",
    # not the raw input "fever") -- older run files predating this field
    # fall back to raw input, which will only match symptoms that happen to
    # be exact KG node names already
    symptoms = run.get("resolved_symptoms") or [s.strip().lower() for s in run.get("symptoms", [])]

    G = nx.Graph()
    node_types = {}

    # add every reported symptom up front, even ones with no edge to any of
    # the shown diseases -- an isolated symptom node is a real, honest
    # signal ("this was reported but didn't match any top disease"), not
    # something to silently drop
    for s in symptoms:
        G.add_node(s); node_types[s] = "symptom"

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for r in session.run(SYMPTOM_QUERY, diseases=diseases, symptoms=symptoms):
            d, s = r["disease"], r["symptom"]
            G.add_node(d); node_types[d] = "disease"
            G.add_node(s); node_types[s] = "symptom"
            G.add_edge(d, s, rel_type="HAS_SYMPTOM")
    driver.close()

    for d in run.get("diseases", []):
        disease = d["disease"]
        G.add_node(disease); node_types[disease] = "disease"
        for drug in d.get("drugs", []):
            G.add_node(drug); node_types[drug] = "drug"
            G.add_edge(drug, disease, rel_type="TREATS")

    interaction_edges = []
    for i in run.get("interactions", []):
        a, b = i.get("d1.name"), i.get("d2.name")
        if a and b:
            G.add_node(a); node_types.setdefault(a, "drug")
            G.add_node(b); node_types.setdefault(b, "drug")
            G.add_edge(a, b, interaction=True, rel_type="INTERACTS_WITH")
            interaction_edges.append((a, b))

    # Okabe-Ito colorblind-safe triple: blue / orange / reddish-purple. The
    # third category originally used bluish-green (#009E73), swapped for
    # #CC79A7 -- still colorblind-safe, reads less like a stray "success
    # green" against the blue/orange pair.
    color_map = {"disease": "#0072B2", "symptom": "#E69F00", "drug": "#CC79A7"}
    colors = [color_map[node_types[n]] for n in G.nodes()]

    # Node size and label wrapping scale together so long names (e.g.
    # "blurred and distorted vision") stay fully inside the circle instead of
    # spilling onto the canvas -- short names get small circles, long ones
    # get bigger circles with the text wrapped across 2-3 lines.
    WRAP_WIDTH = 11
    wrapped_labels = {n: textwrap.fill(n, width=WRAP_WIDTH) for n in G.nodes()}
    node_sizes = []
    for n in G.nodes():
        n_lines = wrapped_labels[n].count("\n") + 1
        longest_line = max(len(line) for line in wrapped_labels[n].split("\n"))
        node_sizes.append(max(1500, 210 * longest_line + 550 * n_lines))

    # Relation type is now encoded by line style, not by a text label on every
    # edge -- with up to 7 HAS_SYMPTOM edges converging on one node, rotated
    # overlapping text labels were unreadable. Line style also survives
    # greyscale printing, which a color-only encoding would not.
    LINESTYLE = {"HAS_SYMPTOM": "solid", "TREATS": "dashed", "INTERACTS_WITH": "dotted"}
    edges_by_type = {rt: [] for rt in LINESTYLE}
    for u, v, dd in G.edges(data=True):
        edges_by_type[dd["rel_type"]].append((u, v))

    pos = nx.spring_layout(G, seed=42, k=0.9)

    plt.figure(figsize=(14, 10))
    for rel_type, style in LINESTYLE.items():
        width = 2 if rel_type == "INTERACTS_WITH" else 1
        nx.draw_networkx_edges(G, pos, edgelist=edges_by_type[rel_type],
                                edge_color="#666666", width=width, style=style)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes, edgecolors="black")

    # Labels wrapped and centered inside each node (white, bold, small) --
    # node size above was chosen specifically to fit this text, so nothing
    # spills outside the circle or floats loose on the canvas.
    nx.draw_networkx_labels(G, pos, labels=wrapped_labels, font_size=7,
                             font_family="Times New Roman", font_color="white",
                             font_weight="bold")

    edge_labels = {(u, v): dd["rel_type"] for u, v, dd in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6.5,
                                  font_color="#333333", font_family="Times New Roman",
                                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=12, label=t)
        for t, c in color_map.items()
    ]
    for rel_type, style in LINESTYLE.items():
        legend_handles.append(plt.Line2D([0], [0], color="#666666", linestyle=style, label=rel_type))
    plt.legend(handles=legend_handles, loc="upper left", prop={"family": "Times New Roman"})

    plt.title(f"Disease -> Symptom / Drug -> Interaction ({run_path.stem})", family="Times New Roman")
    plt.axis("off")
    plt.tight_layout()

    out_path = run_path.with_name(run_path.stem + "_graph.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved to {out_path}")
    print(f"Interaction edges: {len(interaction_edges)}")


if __name__ == "__main__":
    main()
