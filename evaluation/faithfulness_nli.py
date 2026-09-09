import re
import transformers.modeling_utils as _mu
from transformers import AutoModelForSequenceClassification

# Purpose-built for RAG/summarization factual-consistency checking (source
# document vs. generated text), unlike generic MNLI/SNLI entailment models
# which are trained on short synthetic sentence pairs and are miscalibrated
# for long, jargon-dense medical text. Returns one consistency score in
# [0, 1] directly -- 0 = hallucinated, 1 = factually consistent.
HALLUCINATION_MODEL = "vectara/hallucination_evaluation_model"

LOW_CONSISTENCY_THRESHOLD = 0.5

# Defensive per-piece cap. Individual evidence pieces (a single PubMed
# abstract snippet, a single RAG answer) are naturally well under this on
# their own -- this is not the primary truncation mechanism anymore, just a
# safety net against one unusually long piece overflowing the 512-token limit.
MAX_EVIDENCE_PIECE_CHARS = 1500


def _load_hhem_model():
    # HHEM's custom modeling code predates transformers' current tied-weights
    # API (missing `all_tied_weights_keys` crashes on load as of transformers
    # 5.x). The patch below no-ops that check; without it, the model loads
    # but silently leaves t5.transformer.encoder.embed_tokens randomly
    # initialized instead of tied to t5.transformer.shared, making every
    # score non-reproducible garbage (verified: scores for the same input
    # pair varied from 0.0001 to 0.86 across separate loads before this fix).
    # The explicit reassignment below performs the tying transformers would
    # normally do automatically, and was verified to make scores stable
    # and directionally sane across repeated fresh loads.
    orig = _mu.PreTrainedModel.mark_tied_weights_as_initialized

    def _patched(self, loading_info):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return orig(self, loading_info)

    _mu.PreTrainedModel.mark_tied_weights_as_initialized = _patched

    model = AutoModelForSequenceClassification.from_pretrained(
        HALLUCINATION_MODEL, trust_remote_code=True
    )
    model.t5.transformer.encoder.embed_tokens = model.t5.transformer.shared
    return model


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


class FaithfulnessScorer:

    def __init__(self):
        self.model = _load_hhem_model()

    def score_pair(self, premise, hypothesis):
        return float(self.model.predict([(premise, hypothesis)])[0])

    def _score_sentences(self, evidence_pieces, answer_text):
        """Shared core: per-sentence max-score-against-any-evidence-piece.
        Returns (sentences, pieces, scores, best_piece_idx) so callers can
        either aggregate (score_answer) or inspect individual claims
        (score_answer_detailed) from the same underlying computation."""
        sentences = split_sentences(answer_text)
        pieces = [p[:MAX_EVIDENCE_PIECE_CHARS] for p in evidence_pieces if p]

        if not sentences or not pieces:
            return sentences, pieces, [], []

        pairs = [(piece, sentence) for sentence in sentences for piece in pieces]
        raw_scores = [float(s) for s in self.model.predict(pairs)]

        scores = []
        best_piece_idx = []
        idx = 0
        for _ in sentences:
            chunk = raw_scores[idx: idx + len(pieces)]
            best = max(range(len(chunk)), key=lambda i: chunk[i])
            scores.append(chunk[best])
            best_piece_idx.append(best)
            idx += len(pieces)

        return sentences, pieces, scores, best_piece_idx

    def score_answer(self, evidence_pieces, answer_text):
        """
        Claim-level faithfulness: splits the generated answer into sentences
        and scores each sentence against every individual evidence piece
        separately, taking the max -- a claim counts as supported if at
        least one source backs it. This replaces an earlier version that
        concatenated every piece into one premise and truncated it, which
        silently dropped most of the evidence (a disease record with 5
        citations + 3 RAG snippets can total 10k+ characters; truncating to
        ~1500 kept only the first citation, understating faithfulness for
        anything supported by evidence 2-8).
        """
        sentences, pieces, scores, _ = self._score_sentences(evidence_pieces, answer_text)

        if not sentences or not pieces:
            return {"avg_consistency": 0.0, "min_consistency": 0.0, "low_consistency_rate": 0.0, "num_claims": 0}

        n = len(scores)
        return {
            "avg_consistency": sum(scores) / n,
            "min_consistency": min(scores),
            "low_consistency_rate": sum(1 for s in scores if s < LOW_CONSISTENCY_THRESHOLD) / n,
            "num_claims": n,
        }

    def score_answer_detailed(self, evidence_pieces, answer_text):
        """Same computation as score_answer, but returns each sentence with
        its score and best-matching evidence piece -- for inspecting which
        specific claims drive a low aggregate score, not just the aggregate."""
        sentences, pieces, scores, best_piece_idx = self._score_sentences(evidence_pieces, answer_text)

        return [
            {
                "sentence": sentence,
                "score": score,
                "best_evidence": pieces[piece_idx],
            }
            for sentence, score, piece_idx in zip(sentences, scores, best_piece_idx)
        ]
