"""Contrato de entrada e ordem das características finais."""

SEED = 0
TEST_SIZE = 0.20
PREDICTORS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "BMI", "Age"
]
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"
]
PIMA_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "Outcome"
]
ZERO_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
SAMPLING_METHODS = ("none", "smote", "adasyn")
