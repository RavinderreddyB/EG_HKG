# data/

## Committed to git

Original curated inputs, manually-extracted work with no regenerating
script, or third-party data whose license permits redistribution:

- `Disease precaution.csv`
- `raw/symptoms/DiseaseAndSymptoms.csv` — Kaggle disease-symptom dataset (41 diseases, 131 symptoms)
- `raw/symptoms/MedlinePlus_Symptoms.csv`, `raw/symptoms/MedlinePlus_Treatments.csv` — manually extracted and filtered from the raw MedlinePlus XML; not regenerable by any script in this repo
- `raw/drugs/drug_disease.csv` — SIDER-derived drug-indication data (CID -> disease). SIDER is released under CC BY-NC-SA; redistribution here is with attribution, matching that license
- `raw/drugs/cid_to_name.csv` — PubChem CID-to-name lookup table
- `processed/medquad.json`, `processed/medquad_filtered.json` — processed MedQuAD Q&A pairs
- `vector_store/medlineplus_evidence.json` — extracted MedlinePlus evidence corpus

## Not committed (gitignored) — license-restricted, no redistribution

- `raw/drugs/db_drug_interactions.csv` — DrugBank-derived interaction data. DrugBank's academic license does not permit redistribution.
- `raw/drugs/meddra_all_indications.tsv` — SIDER's own export of MedDRA-coded indication terms; kept local rather than risk a MedDRA terminology-licensing conflict.

Both are required to rebuild the drug-interaction and drug-indication edges
(`kg/interaction_builder.py`, `scripts/process_sider.py`) — get your own
copies directly from DrugBank (academic license) and SIDER
(http://sideeffects.embl.de/) if you need to rebuild those edges from
scratch.

## Not committed (gitignored) — third-party, regenerable

Public datasets rebuilt by scripts already in `scripts/`:

| Path | Source | Rebuild command |
|---|---|---|
| `raw/medlineplus/mplus_topics_*.xml` | NIH/NLM MedlinePlus bulk XML feed, https://medlineplus.gov/xml.html | Manual download (no fetch script — NLM does not version this feed) |
| `raw/medquad/medquad.jsonl` | MedQuAD (Ben Abacha & Demner-Fushman) | `python -m scripts.download_medquad` |

After fetching the raw MedlinePlus XML, rebuild the extracted evidence corpus with:
```
python -m scripts.extract_medlineplus_evidence
```

## Also gitignored, not third-party

- `backups/` — raw Neo4j graph dumps (~50MB each). Kept locally only — see
  the root `.gitignore` comment for why these aren't safe to delete outright.
- `vector_store/*.faiss`, `vector_store/medquad_map.json` — FAISS index,
  rebuilt via `python -m scripts.build_vector_index` (over the 100MB GitHub
  limit uncompressed; ~3 hours on CPU).
