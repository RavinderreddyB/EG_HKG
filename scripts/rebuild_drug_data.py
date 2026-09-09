"""
Rebuilds cid_to_name.csv and drug_disease.csv from scratch using real PubChem CIDs.

The original files used unverified 'cid1xxxxxxx' identifiers that do not
resolve to the correct compounds on PubChem (confirmed: cid100005267 was
labeled 'acetaminophen' but real PubChem CID 5267 is a different compound).

This script starts from a curated disease -> standard first-line drug mapping
and resolves each drug name to its real PubChem CID via the name lookup API,
so every entry is traceable and correct.
"""

import csv
import time
import requests
from pathlib import Path

PUBCHEM_CID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
REQUEST_DELAY = 0.25  # PubChem asks for max 5 req/sec

OUT_DIR = Path("data/raw/drugs")
CID_TO_NAME_PATH = OUT_DIR / "cid_to_name.csv"
DRUG_DISEASE_PATH = OUT_DIR / "drug_disease.csv"

# Curated standard first-line treatments for the 41 diseases already in the KG.
# Scope and drug categories match what was already in the original dataset
# (antibiotics, antivirals, antifungals, antimalarials, NSAIDs, etc.)
# This is for a research prototype, not clinical guidance.
DISEASE_DRUGS = {
    "(vertigo) paroymsal  positional vertigo": ["meclizine", "betahistine"],
    "acne": ["clindamycin", "doxycycline", "isotretinoin", "benzoyl peroxide"],
    "aids": ["tenofovir", "emtricitabine", "efavirenz", "dolutegravir"],
    "alcoholic hepatitis": ["prednisolone", "pentoxifylline"],
    "allergy": ["cetirizine", "loratadine", "diphenhydramine"],
    "arthritis": ["ibuprofen", "naproxen", "methotrexate"],
    "bronchial asthma": ["salbutamol", "budesonide", "montelukast"],
    "cervical spondylosis": ["ibuprofen", "naproxen", "cyclobenzaprine"],
    "chicken pox": ["acyclovir", "valacyclovir"],
    "chronic cholestasis": ["ursodeoxycholic acid", "cholestyramine"],
    "common cold": ["paracetamol", "pseudoephedrine", "dextromethorphan"],
    "dengue": ["paracetamol"],
    "diabetes": ["metformin", "insulin", "glimepiride"],
    "dimorphic hemmorhoids(piles)": ["hydrocortisone", "diosmin"],
    "drug reaction": ["diphenhydramine", "prednisolone"],
    "fungal infection": ["fluconazole", "itraconazole", "clotrimazole", "terbinafine"],
    "gastroenteritis": ["ondansetron", "loperamide", "oral rehydration salts"],
    "gerd": ["omeprazole", "pantoprazole", "ranitidine"],
    "heart attack": ["aspirin", "clopidogrel", "atorvastatin", "metoprolol"],
    "hepatitis a": ["paracetamol"],
    "hepatitis b": ["tenofovir", "entecavir"],
    "hepatitis c": ["sofosbuvir", "ledipasvir"],
    "hepatitis d": ["pegylated interferon alfa"],
    "hepatitis e": ["ribavirin"],
    "hypertension": ["lisinopril", "losartan", "amlodipine", "hydrochlorothiazide"],
    "hyperthyroidism": ["methimazole", "propylthiouracil", "propranolol"],
    "hypoglycemia": ["glucose", "glucagon"],
    "hypothyroidism": ["levothyroxine"],
    "impetigo": ["mupirocin", "cephalexin", "amoxicillin clavulanate"],
    "jaundice": ["ursodeoxycholic acid"],
    "malaria": ["artemether", "lumefantrine", "chloroquine", "primaquine"],
    "migraine": ["sumatriptan", "propranolol", "ibuprofen"],
    "osteoarthristis": ["paracetamol", "ibuprofen", "diclofenac"],
    "paralysis (brain hemorrhage)": ["mannitol", "labetalol"],
    "peptic ulcer diseae": ["omeprazole", "amoxicillin", "clarithromycin"],
    "pneumonia": ["amoxicillin", "azithromycin", "ceftriaxone"],
    "psoriasis": ["methotrexate", "calcipotriol", "cyclosporine"],
    "tuberculosis": ["isoniazid", "rifampicin", "ethambutol", "pyrazinamide"],
    "typhoid": ["ciprofloxacin", "ceftriaxone", "azithromycin"],
    "urinary tract infection": ["nitrofurantoin", "ciprofloxacin", "trimethoprim"],
    "varicose veins": ["diosmin", "hesperidin"],
}


def fetch_cid(drug_name, retries=2):
    url = PUBCHEM_CID_URL.format(name=requests.utils.quote(drug_name))
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            cids = r.json().get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
                continue
            print(f"  [WARN] Failed to resolve '{drug_name}': {e}")
            return None


def main():
    all_drugs = sorted({drug for drugs in DISEASE_DRUGS.values() for drug in drugs})
    print(f"Resolving {len(all_drugs)} unique drug names via PubChem...\n")

    name_to_cid = {}
    for i, drug in enumerate(all_drugs):
        cid = fetch_cid(drug)
        if cid:
            name_to_cid[drug] = cid
            print(f"  [{i+1}/{len(all_drugs)}] {drug} -> CID {cid}")
        else:
            print(f"  [{i+1}/{len(all_drugs)}] {drug} -> NOT FOUND (skipped)")
        time.sleep(REQUEST_DELAY)

    unresolved = [d for d in all_drugs if d not in name_to_cid]
    if unresolved:
        print(f"\n[WARN] {len(unresolved)} drug(s) could not be resolved and will be excluded: {unresolved}")

    # Write cid_to_name.csv
    with open(CID_TO_NAME_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cid", "name"])
        for drug, cid in sorted(name_to_cid.items(), key=lambda x: x[1]):
            writer.writerow([cid, drug])

    # Write drug_disease.csv
    with open(DRUG_DISEASE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["drug", "disease"])
        for disease, drugs in DISEASE_DRUGS.items():
            for drug in drugs:
                if drug in name_to_cid:
                    writer.writerow([name_to_cid[drug], disease])

    print(f"\n[OK] Wrote {CID_TO_NAME_PATH} ({len(name_to_cid)} drugs)")
    print(f"[OK] Wrote {DRUG_DISEASE_PATH}")
    print("\nNext steps:")
    print("  1. Wipe Neo4j: MATCH (n) DETACH DELETE n")
    print("  2. Rebuild KG: python -m scripts.build_kg")
    print("  3. Re-run provenance: python -m scripts.add_provenance")


if __name__ == "__main__":
    main()
