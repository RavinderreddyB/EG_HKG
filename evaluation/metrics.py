import re

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "for", "and", "or", "with", "by", "this", "that", "it", "as", "be", "can",
    "may", "if", "your", "you", "which", "from", "has", "have", "had", "not",
    "no", "do", "does", "will", "should", "would", "could", "their", "them",
}


def _tokenize(text):
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _overlap_ratio(a_tokens, b_tokens):
    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def _split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


class MedicalEvaluator:
    """
    Lightweight, dependency-free heuristic metrics for quick end-to-end
    sanity checks (see scripts/run_evaluation.py). These are word-overlap
    proxies, not the rigorous per-claim NLI check in
    evaluation/faithfulness_nli.py -- use that module for the faithfulness
    numbers reported in the writeup. This one exists to catch obvious
    regressions cheaply (no model load) across a handful of test queries.
    """

    def evidence_coverage(self, answer, evidence, overlap_threshold=0.15):
        """Fraction of the given evidence pieces whose vocabulary shows up
        meaningfully in the generated answer -- how much of what was
        retrieved actually made it into the answer."""
        if not evidence or not answer:
            return 0.0

        answer_tokens = _tokenize(answer)
        covered = sum(
            1 for e in evidence
            if _overlap_ratio(_tokenize(e.get("text", "")), answer_tokens) >= overlap_threshold
        )
        return covered / len(evidence)

    def hallucination_score(self, answer, evidence, overlap_threshold=0.1):
        """Fraction of the answer's sentences that share little to no
        vocabulary with ANY provided evidence piece -- a cheap proxy for
        ungrounded content. Word-overlap only, no semantic matching; treat
        as a rough signal, not a substitute for the NLI faithfulness check."""
        if not answer:
            return 0.0

        sentences = _split_sentences(answer)
        if not sentences:
            return 0.0

        evidence_tokens = [_tokenize(e.get("text", "")) for e in evidence]

        ungrounded = 0
        for sentence in sentences:
            sent_tokens = _tokenize(sentence)
            best = max((_overlap_ratio(sent_tokens, ev) for ev in evidence_tokens), default=0.0)
            if best < overlap_threshold:
                ungrounded += 1

        return ungrounded / len(sentences)

    def precision_at_k(self, ranked_evidence, k=3, trust_threshold=0.5):
        """Of the top-k ranked evidence pieces, what fraction the model's
        own TrustRank considers relevant (trust_score >= threshold). This
        is self-referential -- checked against the ranker's own score, not
        an independent gold label -- so it's a sanity check that ranking
        and filtering agree, not an external precision measurement."""
        top_k = ranked_evidence[:k]
        if not top_k:
            return 0.0
        relevant = sum(1 for e in top_k if e.get("trust_score", 0.0) >= trust_threshold)
        return relevant / len(top_k)

    def avg_trust_score(self, ranked_evidence, k=3):
        """Average TrustRank score across the top-k ranked evidence pieces."""
        top_k = ranked_evidence[:k]
        if not top_k:
            return 0.0
        return sum(e.get("trust_score", 0.0) for e in top_k) / len(top_k)
