import pandas as pd
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from tqdm import tqdm

BATCH_SIZE = 500


def normalize(text):
    return str(text).strip().lower()


class InteractionBuilder:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    @staticmethod
    def _batch_tx(tx, batch):
        tx.run("""
            UNWIND $batch AS row
            MERGE (d1:Drug {name: row.d1})
            MERGE (d2:Drug {name: row.d2})
            MERGE (d1)-[r:INTERACTS_WITH]->(d2)
            SET r.description = row.desc
        """, batch=batch)

    def build_from_csv(self, path):
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows")

        rows = []
        for _, row in df.iterrows():
            rows.append({
                "d1": normalize(row["Drug 1"]),
                "d2": normalize(row["Drug 2"]),
                "desc": str(row["Interaction Description"]).strip()
            })

        total = 0
        for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Inserting interactions"):
            batch = rows[i:i + BATCH_SIZE]
            with self.driver.session() as session:
                session.execute_write(self._batch_tx, batch)
            total += len(batch)

        print(f"[OK] Inserted {total} interactions")