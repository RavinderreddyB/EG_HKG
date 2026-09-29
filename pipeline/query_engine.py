from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class SymptomQueryEngine:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def validate_symptoms(self, symptoms):
        """Resolve each raw input term to the KG symptom node(s) it could mean.

        Returns (term_to_nodes, unmatched) where term_to_nodes maps each
        recognized input term to a list of candidate node names -- a term
        can map to more than one node (e.g. "fever" -> ["high fever", "mild
        fever"]) without that ambiguity silently inflating downstream scores,
        since find_diseases groups by term, not by node.
        """
        query = """
        MATCH (s:Symptom)
        WHERE s.name IN $symptoms
           OR ANY(term IN $symptoms WHERE s.name CONTAINS term)
        RETURN s.name AS name
        """
        with self.driver.session() as session:
            all_matches = [r["name"] for r in session.run(query, symptoms=symptoms)]

        term_to_nodes = {}
        unmatched = []

        for term in symptoms:
            if term in all_matches:
                term_to_nodes[term] = [term]
                continue

            hits = [name for name in all_matches if term in name]
            if hits:
                term_to_nodes[term] = hits
            else:
                unmatched.append(term)

        return term_to_nodes, unmatched

    def check_ranking_confidence(self, ranked_results, tie_threshold=3):
        """Flag low-confidence rankings by looking at the ACTUAL ranked
        output rather than guessing from input symptom frequency alone.

        Rationale: an earlier version of this check classified a query as
        "generic" only if every input term individually matched a
        high-frequency symptom node. That approach missed real dilution
        cases -- e.g. "cold, cough, headache", where "cold" resolves to a
        genuinely rare symptom node that happens to match none of the
        actually-competing diseases, so only "cough"/"headache" end up
        doing the real ranking work, and several diseases tie on
        match_count regardless of "cold" being individually rare. Checking
        the ranked output directly for ties/near-ties catches this and any
        other cause of low ranking confidence in one place, rather than
        trying to predict it from the input.

        A "tie" here means sharing the same match_count as the #1 result --
        match_count is the primary, dominant sort key (see find_diseases),
        so diseases tied on it are functionally indistinguishable to the
        ranking formula regardless of secondary score/coverage differences.

        Returns dict with is_low_confidence (bool), tied_diseases (list of
        disease names sharing top match_count), and top_match_count.
        """
        if not ranked_results:
            return {"is_low_confidence": False, "tied_diseases": [], "top_match_count": 0}

        top_match_count = ranked_results[0]["match_count"]
        tied = [r["disease"] for r in ranked_results if r["match_count"] == top_match_count]

        return {
            "is_low_confidence": len(tied) >= tie_threshold,
            "tied_diseases": tied,
            "top_match_count": top_match_count,
        }

    def find_diseases(self, term_to_nodes):

        node_to_term = {
            node: term
            for term, nodes in term_to_nodes.items()
            for node in nodes
        }
        all_nodes = list(node_to_term.keys())

        query = """
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
WHERE s.name IN $symptoms

// frequency
MATCH (s)<-[:HAS_SYMPTOM]-(d2:Disease)
WITH d, s, COUNT(d2) AS disease_freq

// clique
OPTIONAL MATCH (s)-[r:CO_OCCURS_WITH]->(s2:Symptom)
WHERE s2.name IN $symptoms

WITH d, s,
     (1.0 / disease_freq) AS idf,
     COALESCE(r.weight, 1) AS co_weight

WITH d, s, SUM(idf * co_weight) AS node_score

WITH d, COLLECT({node: s.name, node_score: node_score}) AS node_scores

// total symptoms per disease
MATCH (d)-[:HAS_SYMPTOM]->(all_s:Symptom)
WITH d, node_scores, COUNT(all_s) AS total_symptoms

RETURN d.name AS disease,
       node_scores,
       total_symptoms
"""

        with self.driver.session() as session:
            records = session.run(query, symptoms=all_nodes)

            results = []
            for record in records:
                disease = record["disease"]
                total_symptoms = record["total_symptoms"]

                # one score per input TERM, not per matched node -- an
                # ambiguous term (multiple candidate nodes) only counts once,
                # taking its best-matching node's score
                term_scores = {}
                for ns in record["node_scores"]:
                    term = node_to_term[ns["node"]]
                    term_scores[term] = max(term_scores.get(term, 0.0), ns["node_score"])

                match_count = len(term_scores)
                score = sum(term_scores.values())
                coverage = match_count / total_symptoms if total_symptoms else 0.0

                results.append({
                    "disease": disease,
                    "match_count": match_count,
                    "total_symptoms": total_symptoms,
                    "score": score,
                    "coverage": coverage,
                })

        # match_count is the primary sort key so a disease matching more of
        # the reported symptoms always outranks one matching fewer -- neither
        # a long total-symptom list (old bug: narrow diseases like chicken
        # pox outranking common cold) nor a single rare/high-idf symptom
        # (reverted attempt: paralysis/heart attack outranking a disease
        # matching all 3 reported symptoms) can invert that. score and
        # coverage only break ties within the same match_count.
        results.sort(key=lambda r: (-r["match_count"], -r["score"], -r["coverage"]))
        return results[:10]