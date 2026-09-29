import os
import time
import requests
import networkx as nx
from tqdm import tqdm

ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

NCBI_API_KEY = os.getenv("NCBI_API_KEY", None)
REQUEST_DELAY = 0.11 if NCBI_API_KEY else 0.35


def fetch_citing_pmids(pmid):
    """PMIDs of PMC-indexed papers that cite the given PMID (NCBI elink)."""
    params = {
        "dbfrom": "pubmed",
        "db": "pubmed",
        "id": pmid,
        "linkname": "pubmed_pubmed_citedin",
        "retmode": "json",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    try:
        r = requests.get(ELINK_URL, params=params, timeout=10)
        r.raise_for_status()
        linksets = r.json().get("linksets", [])
        if not linksets:
            return []
        for linksetdb in linksets[0].get("linksetdbs", []):
            if linksetdb.get("linkname") == "pubmed_pubmed_citedin":
                return [str(x) for x in linksetdb.get("links", [])]
        return []
    except Exception:
        return []


def build_citation_graph(pmids):
    """
    Directed citation graph restricted to the given PMID set: edge
    citing_pmid -> cited_pmid exists only if both papers are already
    part of our enriched evidence corpus. External citations (papers
    outside the corpus) are not tracked — this measures authority
    *within* the evidence we actually cite, not global impact.
    """
    pmid_set = set(pmids)
    graph = nx.DiGraph()
    graph.add_nodes_from(pmid_set)

    for pmid in tqdm(sorted(pmid_set), desc="Fetching citations"):
        for citer in fetch_citing_pmids(pmid):
            if citer in pmid_set and citer != pmid:
                graph.add_edge(citer, pmid)
        time.sleep(REQUEST_DELAY)

    return graph


def compute_trust_scores(graph):
    """PageRank over the citation graph. Nodes with no in-corpus citations
    still get the uniform baseline score, not zero."""
    if graph.number_of_nodes() == 0:
        return {}
    return nx.pagerank(graph, alpha=0.85)
