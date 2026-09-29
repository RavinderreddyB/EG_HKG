import pandas as pd
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class DrugKGBuilder:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def insert_drug_disease(self, drug, disease):
        with self.driver.session() as session:
            session.execute_write(self._tx, drug, disease)

    @staticmethod
    def _tx(tx, drug, disease):

        tx.run("""
            MERGE (dr:Drug {name: $drug})
        """, drug=drug)

        tx.run("""
            MATCH (d:Disease {name: $disease})
            MATCH (dr:Drug {name: $drug})
            MERGE (d)-[:TREATED_BY]->(dr)
        """, disease=disease, drug=drug)

    def build_from_csv(self, path):

        df = pd.read_csv(path)

        print(f"Loaded {len(df)} rows")

        success = 0

        for _, row in df.iterrows():

            disease = str(row["disease"]).strip()
            drug = str(row["drug"]).strip()

            try:
                self.insert_drug_disease(drug, disease)
                success += 1
            except Exception:
                pass

        print(f"[OK] Inserted {success} relationships")