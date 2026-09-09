from pipeline.full_pipeline import MedicalPipeline

if __name__ == "__main__":
    pipeline = MedicalPipeline()

    user_input = input("Enter symptoms (comma separated): ")
    symptoms = [s.strip().lower() for s in user_input.split(",")]



    pipeline.run(symptoms)

    pipeline.close()