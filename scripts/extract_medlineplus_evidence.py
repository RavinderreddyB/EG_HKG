"""Extract per-disease full-text evidence from the raw MedlinePlus XML dump
for the 84 diseases added via the MedlinePlus expansion. This is real
disease-specific prose (not the terse treatment_modalities slugs used for
the KG), used as RAG grounding + faithfulness-eval evidence for these
diseases, replacing the MedQuAD index (which has near-zero real coverage
for these practical/common conditions -- confirmed via direct check on
"Dislocated Shoulder": zero true matches, only coincidental keyword hits
on unrelated rare genetic-disorder passages).
"""
import csv
import html
import json
import re
from pathlib import Path

XML_PATH = Path("data/raw/medlineplus/mplus_topics_2026-07-30.xml")
DISEASES_CSV = Path("data/raw/symptoms/MedlinePlus_Symptoms.csv")
OUTPUT_PATH = Path("data/vector_store/medlineplus_evidence.json")

# manual overrides where our disease-name string doesn't exactly match the
# XML topic title (same category of mismatch as the earlier Crohns/CFS bug)
NAME_OVERRIDES = {
    "Myalgic Encephalomyelitis Chronic Fatigue Syndrome": "Myalgic Encephalomyelitis/Chronic Fatigue Syndrome",
    "Crohns Disease": "Crohn's Disease",
}


def load_target_diseases():
    with open(DISEASES_CSV, encoding="utf-8") as f:
        return [row["Disease"] for row in csv.DictReader(f)]


def load_topic_blocks():
    with open(XML_PATH, encoding="utf-8") as f:
        content = f.read()
    return re.findall(r"<health-topic .*?</health-topic>", content, re.DOTALL)


def strip_html(raw):
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def main():
    diseases = load_target_diseases()
    blocks = load_topic_blocks()

    # title (lowercased) -> best block (the one with the longest full-summary,
    # to prefer the main topic page over short duplicate/related entries)
    by_title = {}
    for block in blocks:
        m = re.search(r'<health-topic[^>]*title="([^"]*)"', block)
        if not m:
            continue
        title = m.group(1).strip().lower()
        summary_m = re.search(r"<full-summary>(.*?)</full-summary>", block, re.DOTALL)
        summary_len = len(summary_m.group(1)) if summary_m else 0
        if title not in by_title or summary_len > by_title[title][1]:
            by_title[title] = (block, summary_len)

    evidence = {}
    unmatched = []

    for disease in diseases:
        lookup_name = NAME_OVERRIDES.get(disease, disease)
        entry = by_title.get(lookup_name.lower())
        if not entry:
            unmatched.append(disease)
            continue

        block, _ = entry
        summary_m = re.search(r"<full-summary>(.*?)</full-summary>", block, re.DOTALL)
        if not summary_m:
            unmatched.append(disease)
            continue

        clean_text = strip_html(summary_m.group(1))
        if not clean_text:
            unmatched.append(disease)
            continue

        url_m = re.search(r'url="([^"]*)"', block)
        evidence[disease] = {
            "text": clean_text,
            "url": url_m.group(1) if url_m else None,
        }

    print(f"Target diseases: {len(diseases)}")
    print(f"Extracted: {len(evidence)}")
    print(f"Unmatched: {len(unmatched)}")
    if unmatched:
        for d in unmatched:
            print(f"  - {d}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
