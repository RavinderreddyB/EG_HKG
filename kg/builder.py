# kg/builder.py

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class MedicalKGBuilder:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    # ---------------------------
    # Constraints
    # ---------------------------
    def create_constraints(self):
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT disease_name IF NOT EXISTS
                FOR (d:Disease) REQUIRE d.name IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT symptom_name IF NOT EXISTS
                FOR (s:Symptom) REQUIRE s.name IS UNIQUE
            """)
            session.run("""
                CREATE INDEX disease_name_idx IF NOT EXISTS
                FOR (d:Disease) ON (d.name)
            """)
            session.run("""
                CREATE INDEX symptom_name_idx IF NOT EXISTS
                FOR (s:Symptom) ON (s.name)
            """)
            session.run("""
                CREATE INDEX drug_name_idx IF NOT EXISTS
                FOR (dr:Drug) ON (dr.name)
            """)
        print("[OK] Constraints and indexes created")

    # ---------------------------
    # Insert Data
    # ---------------------------
    def insert_data(self, disease, symptoms, source="DiseaseAndSymptoms"):
        with self.driver.session() as session:
            session.execute_write(self._insert_tx, disease, symptoms, source)

    @staticmethod
    def _insert_tx(tx, disease, symptoms, source):

        # Create Disease
        tx.run(
            """
            MERGE (d:Disease {name: $disease})
            SET d.source = $source, d.confidence = $confidence
            """,
            disease=disease,
            source=source,
            confidence=0.6
        )

        # Create Symptoms + Relations
        for symptom in symptoms:
            tx.run(
                """
                MERGE (s:Symptom {name: $symptom})
                WITH s
                MATCH (d:Disease {name: $disease})
                MERGE (d)-[:HAS_SYMPTOM]->(s)
                """,
                symptom=symptom,
                disease=disease
            )

        # Clique (Symptom co-occurrence)
        for i in range(len(symptoms)):
            for j in range(i + 1, len(symptoms)):
                tx.run(
                """
                MATCH (s1:Symptom {name: $s1})
                MATCH (s2:Symptom {name: $s2})
                MERGE (s1)-[r:CO_OCCURS_WITH]->(s2)
                ON CREATE SET r.weight = 1
                ON MATCH SET r.weight = r.weight + 1
                """,
                s1=symptoms[i],
                s2=symptoms[j]
            )

    # ---------------------------
    # Build from CSV
    # ---------------------------
    def build_from_csv(self, path, source="DiseaseAndSymptoms"):
        df = pd.read_csv(path)

        print(f"Loaded {len(df)} rows")

        for _, row in tqdm(df.iterrows(), total=len(df)):

            disease = str(row["Disease"]).strip().lower()

            symptoms = []
            for col in df.columns:
                if "Symptom" in col:
                    val = row[col]
                    if pd.notna(val):
                        clean = str(val).strip().replace("_", " ").lower()
                        symptoms.append(clean)

            symptoms = list(set(symptoms))

            if disease and symptoms:
                try:
                    self.insert_data(disease, symptoms, source=source)
                except Exception as e:
                    print(f"Error: {e}")

        print("[OK] KG Build Complete")

    # ---------------------------
    # Build Treatment Modalities (separate from Drug/TREATS —
    # e.g. "chemotherapy", "psychotherapy": named approaches, not
    # specific drugs, so they must never enter drug-interaction checks)
    # ---------------------------
    def build_treatment_modalities(self, path, source="MedlinePlus"):
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} treatment-modality rows")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            disease = str(row["Disease"]).strip().lower()

            treatments = []
            for col in df.columns:
                if "Treatment" in col:
                    val = row[col]
                    if pd.notna(val):
                        clean = str(val).strip().replace("_", " ").lower()
                        treatments.append(clean)

            if disease and treatments:
                try:
                    self._insert_treatments(disease, treatments, source)
                except Exception as e:
                    print(f"Error: {e}")

        print("[OK] Treatment modalities loaded")

    def _insert_treatments(self, disease, treatments, source):
        with self.driver.session() as session:
            session.execute_write(self._insert_treatments_tx, disease, treatments, source)

    @staticmethod
    def _insert_treatments_tx(tx, disease, treatments, source):
        for treatment in treatments:
            tx.run(
                """
                MERGE (t:Treatment {name: $treatment})
                SET t.source = $source
                WITH t
                MATCH (d:Disease {name: $disease})
                MERGE (d)-[:HAS_TREATMENT]->(t)
                """,
                treatment=treatment,
                disease=disease,
                source=source
            )

    # ---------------------------
    # Build Drug-Disease Links
    # ---------------------------
    def build_drug_relationships(self, path):
        df = pd.read_csv(path)
        print(f"Loading {len(df)} drug-disease relationships")

        batch = []
        batch_size = 500

        for _, row in tqdm(df.iterrows(), total=len(df)):
            drug_cid = str(row["drug"]).strip()
            disease = str(row["disease"]).strip().lower()

            if drug_cid and disease:
                batch.append({"drug_cid": drug_cid, "disease": disease})

            if len(batch) >= batch_size:
                self._batch_drug_tx(batch)
                batch = []

        if batch:
            self._batch_drug_tx(batch)

        print("[OK] Drug-disease relationships loaded")

    def _batch_drug_tx(self, batch):
        with self.driver.session() as session:
            session.execute_write(self._batch_drug_write, batch)

    @staticmethod
    def _batch_drug_write(tx, batch):
        tx.run("""
            UNWIND $batch AS row
            MERGE (drug:Drug {name: row.drug_cid})
            WITH drug, row
            MATCH (d:Disease {name: row.disease})
            MERGE (drug)-[:TREATS]->(d)
        """, batch=batch)