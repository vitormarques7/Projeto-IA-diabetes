"""KNN de referência sobre o pré-processamento compartilhado."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from src.data import load_split
from src.data.schema import SAMPLING_METHODS
from src.preprocessing import build_pipeline


REFERENCE_PARAMS = {"n_neighbors": 95, "weights": "distance", "metric": "manhattan"}


def evaluate(model, X, y):
    """Métricas da classe positiva; AUC usa probabilidades."""
    predicted = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "auc": roc_auc_score(y, probabilities),
    }


def main():
    split = load_split()
    rows = []
    for scenario in SAMPLING_METHODS:
        model = build_pipeline(KNeighborsClassifier(**REFERENCE_PARAMS), scenario)
        model.fit(split.X_train, split.y_train)
        rows.append({"cenario": scenario, **evaluate(model, split.X_test, split.y_test)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
