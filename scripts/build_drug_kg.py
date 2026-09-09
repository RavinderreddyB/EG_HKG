from kg.drug_builder import DrugKGBuilder

CSV_PATH = "data/raw/drugs/drug_disease.csv"

if __name__ == "__main__":
    builder = DrugKGBuilder()
    builder.build_from_csv(CSV_PATH)
    builder.close()