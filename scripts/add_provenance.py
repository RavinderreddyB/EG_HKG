import json

import os
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from neo4j import GraphDatabase
from kg.pubmed_linker import PubMedLinker
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

RESULTS_PATH  = Path("logs/provenance_results.json")
PROGRESS_PATH = Path("logs/provenance_progress.json")
CHECKPOINT_EVERY = 50

NCBI_API_KEY = os.getenv("NCBI_API_KEY", None)
MAX_WORKERS = 3 if NCBI_API_KEY else 1

results_lock = threading.Lock()
done_lock = threading.Lock()


def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return set(json.load(f).get("done", []))
    return set()


def save_progress(done_keys):
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"done": list(done_keys)}, f)


def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def fetch_all_pairs(driver):
    query = """
    MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
    WHERE r.evidence_src IS NULL OR r.evidence_src = 'DiseaseAndSymptoms'
    RETURN d.name AS disease, s.name AS symptom
    """
    with driver.session() as session:
        return [record.data() for record in session.run(query)]


def write_provenance(driver, disease, symptom, provenance):
    query = """
    MATCH (d:Disease {name: $disease})-[r:HAS_SYMPTOM]->(s:Symptom {name: $symptom})
    SET r.pmid             = $pmid,
        r.pub_title        = $pub_title,
        r.pub_year         = $pub_year,
        r.abstract_snippet = $abstract_snippet,
        r.evidence_src     = $evidence_src,
        r.confidence       = $confidence,
        r.relevance_score  = $relevance_score
    """
    with driver.session() as session:
        session.run(query, disease=disease, symptom=symptom, **provenance)


def write_fallback(driver, disease, symptom):
    query = """
    MATCH (d:Disease {name: $disease})-[r:HAS_SYMPTOM]->(s:Symptom {name: $symptom})
    SET r.evidence_src = 'DiseaseAndSymptoms_CSV',
        r.confidence   = 0.50
    """
    with driver.session() as session:
        session.run(query, disease=disease, symptom=symptom)


def process_pair(pair, linker, driver, pbar):
    disease = pair["disease"]
    symptom = pair["symptom"]
    key = f"{disease}|{symptom}"

    result_item = {"disease": disease, "symptom": symptom, "key": key}

    try:
        result = linker.find_best_paper(disease, symptom)

        if result:
            write_provenance(driver, disease, symptom, result)
            result_item.update({"status": "enriched", **result})
            pbar.update(1)
            tqdm.write(f"  {disease} -> {symptom} | PMID {result['pmid']} | conf {result['confidence']}")
        else:
            write_fallback(driver, disease, symptom)
            result_item.update({"status": "fallback"})
            pbar.update(1)
            tqdm.write(f"  {disease} -> {symptom} | no paper found -> fallback")

    except Exception as e:
        write_fallback(driver, disease, symptom)
        result_item.update({"status": "error", "error": str(e)[:100]})
        pbar.update(1)
        tqdm.write(f"  ERROR {disease} -> {symptom}: {str(e)[:80]}")

    return result_item


def main():
    Path("logs").mkdir(exist_ok=True)

    driver  = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    linker  = PubMedLinker()
    done    = load_progress()
    results = load_results()

    pairs = fetch_all_pairs(driver)
    total = len(pairs)
    print(f"Total pairs to enrich: {total}")
    print(f"Already done: {len(done)}")

    pending = [p for p in pairs if f"{p['disease']}|{p['symptom']}" not in done]
    print(f"Remaining: {len(pending)}")
    print(f"Workers: {MAX_WORKERS} (NCBI_API_KEY: {'set' if NCBI_API_KEY else 'not set'})\n")

    enriched  = 0
    fallback  = 0
    errors    = 0
    since_checkpoint = 0
    processed_items = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_pair, pair, linker, driver, tqdm(total=0)): pair
            for pair in pending
        }

        with tqdm(total=len(pending), desc="Enriching") as pbar:
            for future in as_completed(futures):
                result_item = future.result()
                processed_items.append(result_item)

                status = result_item.get("status")
                if status == "enriched":
                    enriched += 1
                else:
                    fallback += 1
                    if status == "error":
                        errors += 1

                done.add(result_item["key"])
                since_checkpoint += 1
                pbar.update(1)

                if since_checkpoint >= CHECKPOINT_EVERY:
                    save_progress(done)
                    save_results(results + processed_items)
                    since_checkpoint = 0
                    pbar.write(f"  [checkpoint saved — {len(done)} done]")

    results.extend(processed_items)
    save_progress(done)
    save_results(results)
    driver.close()

    print(f"\nDone.")
    print(f"  Enriched with PubMed : {enriched}")
    print(f"  Fallback (CSV)       : {fallback - errors}")
    print(f"  Errors               : {errors}")
    print(f"  Coverage rate        : {enriched / total * 100:.1f}%")


if __name__ == "__main__":
    main()
