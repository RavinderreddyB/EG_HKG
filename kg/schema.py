# kg/schema.py

# ============================
# NODE LABELS
# ============================

DISEASE = "Disease"
SYMPTOM = "Symptom"

# ============================
# RELATIONSHIPS
# ============================

HAS_SYMPTOM = "HAS_SYMPTOM"
CO_OCCURS_WITH = "CO_OCCURS_WITH"

# ============================
# METADATA (for future use)
# ============================

SOURCE = "DiseaseAndSymptoms"
DEFAULT_CONFIDENCE = 0.6
DEFAULT_WEIGHT = 1.0

# ============================
# NODE LABELS
# ============================

DRUG = "Drug"

# ============================
# RELATIONSHIPS
# ============================

TREATED_BY = "TREATED_BY"
INTERACTS_WITH = "INTERACTS_WITH"