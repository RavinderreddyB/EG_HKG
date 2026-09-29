import pandas as pd

INPUT_PATH = "data/raw/drugs/meddra_all_indications.tsv"
OUTPUT_PATH = "data/raw/drugs/drug_disease.csv"

df = pd.read_csv(INPUT_PATH, sep="\t", header=None)

print(f"Detected {df.shape[1]} columns")

# Assign columns
df.columns = [
    "drug_id", "umls1", "source", "raw_name",
    "term_type", "umls2", "disease"
]

# [OK] Filter clean data
df = df[
    (df["source"] == "NLP_indication") &
    (df["term_type"] == "PT")
]

# Keep only required columns
df = df[["drug_id", "disease"]]

# Rename
df.columns = ["drug", "disease"]

# Clean text
df["drug"] = df["drug"].astype(str).str.lower()
df["disease"] = df["disease"].astype(str).str.lower()

# Remove duplicates
df = df.drop_duplicates()

# Save
df.to_csv(OUTPUT_PATH, index=False)

print("[OK] Clean drug_disease.csv created")
print(df.head())