"""Experimentos KNN de referência e ajustados, com seleção restrita ao treino."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from src.data import load_split
from src.data.schema import SAMPLING_METHODS, SEED
from src.preprocessing import build_pipeline
from src.utils.paths import RAW_DIR


REFERENCE_PARAMS = {"n_neighbors": 95, "weights": "distance", "metric": "manhattan"}
REFERENCE_URL = (
    "https://github.com/tansin-nabil/Diabetes-Prediction-Using-Machine-Learning/"
    "blob/23b982a20815c4965da47b4b1745c5d3ef0735d1/Clean_work_PIMA%2BRTML.ipynb"
)
# Lista de grades unitárias fixa a ordem dos empates explicitamente.
DEFAULT_GRID = [
    {"model__n_neighbors": [k], "model__weights": [weight], "model__p": [p]}
    for k in [3, 5, 7, 11, 15, 21, 31, 51, 74, 95]
    for weight in ["uniform", "distance"]
    for p in [1, 2]
]


def evaluate(model, X, y):
    """Métricas da classe positiva; AUC usa probabilidades."""
    predicted = model.predict(X)
    positive = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive]
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "auc": roc_auc_score(y, probabilities),
    }


def train_models(X_train, y_train, parameter_grid=None, reference_params=None):
    """Seleciona modelos sem receber dados de teste. Grades menores servem aos testes."""
    grid = DEFAULT_GRID if parameter_grid is None else parameter_grid
    reference = REFERENCE_PARAMS if reference_params is None else reference_params
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fitted, searches, scores = {}, {}, {}
    for scenario in SAMPLING_METHODS:
        print(f"Ajustando KNN: {scenario}", flush=True)
        search = GridSearchCV(
            build_pipeline(KNeighborsClassifier(metric="minkowski"), scenario),
            grid, scoring="roc_auc", cv=cv, n_jobs=1, error_score="raise",
        )
        search.fit(X_train, y_train)
        searches[scenario] = search
        fitted[(scenario, "ajustado")] = search.best_estimator_
        scores[(scenario, "ajustado")] = (
            float(search.best_score_),
            float(search.cv_results_["std_test_score"][search.best_index_]),
        )
        baseline = build_pipeline(KNeighborsClassifier(**reference), scenario)
        # Mesmo conjunto de folds da busca, com todas as etapas reajustadas.
        reference_scores = cross_val_score(
            baseline, X_train, y_train, scoring="roc_auc", cv=cv,
            n_jobs=1, error_score="raise",
        )
        baseline.fit(X_train, y_train)
        fitted[(scenario, "referencia")] = baseline
        scores[(scenario, "referencia")] = (
            float(reference_scores.mean()), float(reference_scores.std())
        )
    # max mantém o primeiro cenário em empates; GridSearchCV faz o mesmo na grade.
    winner = max(SAMPLING_METHODS, key=lambda s: scores[(s, "ajustado")][0])
    return fitted, searches, scores, winner


def evaluate_models(fitted, scores, X_test, y_test):
    """Avalia seis modelos depois da escolha definitiva pelo treino."""
    rows = []
    for scenario in SAMPLING_METHODS:
        for kind in ["referencia", "ajustado"]:
            key = (scenario, kind)
            model = fitted[key]
            rows.append({
                "cenario": scenario, "modelo": kind,
                "cv_auc": scores[key][0], "cv_auc_desvio": scores[key][1],
                **{f"teste_{k}": v for k, v in evaluate(model, X_test, y_test).items()},
                "parametros": json.dumps(model.named_steps["model"].get_params(), sort_keys=True),
            })
    return pd.DataFrame(rows)


def run(raw_dir=RAW_DIR):
    split = load_split(raw_dir)
    fitted, searches, scores, winner = train_models(split.X_train, split.y_train)
    print(f"Cenário selecionado pela validação: {winner}", flush=True)
    table = evaluate_models(fitted, scores, split.X_test, split.y_test)
    if not np.isfinite(table.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("As métricas devem ser finitas")
    print(table.drop(columns="parametros").to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    run(args.raw_dir)


if __name__ == "__main__":
    main()
