from kg.builder import MedicalKGBuilder

SYMPTOMS_CSV = "data/raw/symptoms/DiseaseAndSymptoms.csv"
DRUG_DISEASE_CSV = "data/raw/drugs/drug_disease.csv"

if __name__ == "__main__":
    builder = MedicalKGBuilder()
    builder.create_constraints()
    builder.build_from_csv(SYMPTOMS_CSV)
    builder.build_drug_relationships(DRUG_DISEASE_CSV)
    builder.close()