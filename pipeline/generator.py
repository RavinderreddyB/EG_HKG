from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

MODEL_FALLBACKS = [
    "gemini-3.5-flash-lite",  # primary
    "gemini-3.5-flash",       # fallback
    "gemini-2.5-flash",
    # gemini-2.5-flash-lite removed: confirmed dead (HTTP 404, "no longer
    # available to new users" as of 2026-08, deprecated by Google)
]

REQUEST_TIMEOUT_MS = 30_000


def build_grounded_prompt(disease, drugs, evidence, citations=None, treatment_modalities=None):
    """Shared prompt text for the KG-grounded condition."""

    evidence_text = "\n".join([e["text"] for e in evidence[:5]])

    if drugs:
        drugs_line = ', '.join(drugs)
    elif treatment_modalities:
        drugs_line = (
            f"No specific drug names in knowledge base. General treatment approaches "
            f"only (from MedlinePlus): {', '.join(treatment_modalities)}. "
            f"Describe these approaches only — do not name specific drugs not listed above."
        )
    else:
        drugs_line = (
            "No treatment data available in knowledge base for this disease. "
            "State plainly that no treatment data is available rather than guessing."
        )

    citations_text = ""
    if citations:
        lines = []
        for c in citations:
            cred = c.get("source_credibility")
            cred_str = f", source credibility {cred:.4f}" if cred is not None else ""
            lines.append(f"- {c['symptom']}: PMID {c['pmid']} ({c['year']}), confidence {c['confidence']:.2f}{cred_str} — {c['title']}")
        citations_text = "\n".join(lines)

    return f"""
You are a medical assistant.

ONLY use the provided evidence.
DO NOT hallucinate.

Disease: {disease}

Drugs: {drugs_line}

Evidence:
{evidence_text}

PubMed Citations (cite these as (PMID: xxxxx, Year) in your answer where relevant):
{citations_text if citations_text else "No PubMed citations available"}

Generate:
1. Brief explanation of disease
2. Recommended treatment
3. Safety note if uncertainty exists

Cite PubMed sources inline using the format (PMID: xxxxx, Year) whenever you reference the provided citations.
"""


def build_ungrounded_prompt(disease):
    """Shared prompt text for the no-KG baseline condition."""
    return f"""
You are a medical assistant. Answer using your own medical knowledge.

Disease: {disease}

Generate:
1. Brief explanation of disease
2. Recommended treatment
3. Safety note if uncertainty exists
"""


class MedicalGenerator:

    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def generate(self, disease, drugs, evidence, citations=None, treatment_modalities=None):
        """Returns (answer_text, model_name) -- model_name records which
        entry in MODEL_FALLBACKS actually produced this answer, since
        earlier evaluation runs silently discarded this and could not
        rule out a mix of models within one reported result."""

        prompt = build_grounded_prompt(disease, drugs, evidence, citations, treatment_modalities)

        last_error = None
        for model_name in MODEL_FALLBACKS:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
                    )
                )
                return response.text, model_name
            except (ClientError, ServerError) as e:
                last_error = e
                continue

        raise last_error

    def generate_ungrounded(self, disease):
        """No-KG baseline: pure parametric
        generation, no evidence/citations/drugs supplied, no "only use the
        provided evidence" constraint -- this is what the LLM alone
        produces, to compare against the KG-grounded generate() above.
        Returns (answer_text, model_name), same reasoning as generate()."""

        prompt = build_ungrounded_prompt(disease)

        last_error = None
        for model_name in MODEL_FALLBACKS:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
                    )
                )
                return response.text, model_name
            except (ClientError, ServerError) as e:
                last_error = e
                continue

        raise last_error
