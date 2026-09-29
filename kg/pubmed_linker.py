import time
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config.settings import EMBEDDING_MODEL

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Free tier: 3 req/sec. Set NCBI_API_KEY in .env for 10/sec.
import os
NCBI_API_KEY = os.getenv("NCBI_API_KEY", None)
REQUEST_DELAY = 0.11 if NCBI_API_KEY else 0.35


class PubMedLinker:

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def _search_pmids(self, query, retmax=5):
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        try:
            r = requests.get(PUBMED_SEARCH_URL, params=params, timeout=10)
            r.raise_for_status()
            return r.json()["esearchresult"].get("idlist", [])
        except Exception:
            return []

    def _fetch_summaries(self, pmids):
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        try:
            r = requests.get(PUBMED_SUMMARY_URL, params=params, timeout=10)
            r.raise_for_status()
            result = r.json().get("result", {})
            papers = []
            for pmid in pmids:
                item = result.get(pmid, {})
                if not item or item.get("error"):
                    continue
                papers.append({
                    "pmid": pmid,
                    "title": item.get("title", ""),
                    "year": item.get("pubdate", "")[:4],
                    "source": item.get("source", ""),
                })
            return papers
        except Exception:
            return []

    def _fetch_abstracts_batch(self, pmids):
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        try:
            r = requests.get(PUBMED_FETCH_URL, params=params, timeout=15)
            r.raise_for_status()
            xml = r.text

            abstracts = {}
            import re
            articles = re.split(r"<PubmedArticle>", xml)[1:]
            for article in articles:
                pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
                abstract_match = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
                if pmid_match:
                    pmid = pmid_match.group(1)
                    abstract_text = " ".join(abstract_match).strip() if abstract_match else ""
                    abstracts[pmid] = abstract_text
            return abstracts
        except Exception:
            return {}

    def _score_paper(self, paper, abstract, disease, symptom):
        title = paper["title"].lower()
        abstract_lower = abstract.lower()
        disease_l = disease.lower()
        symptom_l = symptom.lower()

        score = 0.0
        score += 0.35 if disease_l in title else 0.0
        score += 0.25 if symptom_l in title else 0.0
        score += 0.20 if (disease_l in abstract_lower and symptom_l in abstract_lower) else 0.0

        try:
            year = int(paper.get("year", "0"))
            score += 0.10 if year >= 2015 else 0.0
        except ValueError:
            pass

        score += 0.10 if paper.get("source", "") else 0.0

        return score

    def _compute_confidence(self, disease, symptom, abstract):
        if not abstract:
            return 0.50

        query_vec = self.model.encode([f"{disease} {symptom}"])
        abstract_vec = self.model.encode([abstract[:512]])
        sim = cosine_similarity(query_vec, abstract_vec)[0][0]
        return round(float(sim), 4)

    def find_best_paper(self, disease, symptom):
        query = f"{disease} {symptom} clinical"
        time.sleep(REQUEST_DELAY)

        pmids = self._search_pmids(query)
        if not pmids:
            return None

        time.sleep(REQUEST_DELAY)
        papers = self._fetch_summaries(pmids)
        if not papers:
            return None

        time.sleep(REQUEST_DELAY)
        abstracts = self._fetch_abstracts_batch([p["pmid"] for p in papers])

        best_paper = None
        best_score = -1
        best_abstract = ""

        for paper in papers:
            abstract = abstracts.get(paper["pmid"], "")
            score = self._score_paper(paper, abstract, disease, symptom)

            if score > best_score:
                best_score = score
                best_paper = paper
                best_abstract = abstract

        if best_paper is None:
            return None

        confidence = self._compute_confidence(disease, symptom, best_abstract)
        snippet = best_abstract[:300].replace("\n", " ").strip()

        return {
            "pmid": best_paper["pmid"],
            "pub_title": best_paper["title"],
            "pub_year": best_paper["year"],
            "abstract_snippet": snippet,
            "evidence_src": "PubMed",
            "confidence": confidence,
            "relevance_score": round(best_score, 4),
        }
