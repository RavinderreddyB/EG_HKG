from pipeline.query_engine import SymptomQueryEngine

if __name__ == "__main__":
    engine = SymptomQueryEngine()

    symptoms = ["cough", "cold", "fever"]

    results = engine.find_diseases(symptoms)

    print("\nQuery:", symptoms)
    print("\nResults:")

    for r in results:
        print(r)

    engine.close()