from kg.interaction_builder import InteractionBuilder

CSV_PATH = "data/raw/drugs/db_drug_interactions.csv"

if __name__ == "__main__":
    builder = InteractionBuilder()
    builder.build_from_csv(CSV_PATH)
    builder.close()