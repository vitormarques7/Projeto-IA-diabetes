"""Regressão Logística: baseline e versão ajustada por validação cruzada."""


from pathlib import Path


import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


RAIZ = Path(__file__).resolve().parents[2]
PASTA = RAIZ / "data" / "processed"
CENARIOS = ["none", "smote", "adasyn"]
COLUNA_ALVO = "Outcome"
PARAMETROS_BASE = {"max_iter": 1000, "random_state": 42}




def carregar(caminho):
    df = pd.read_csv(caminho)
    return df.drop(columns=[COLUNA_ALVO]), df[COLUNA_ALVO]




def metricas(y_real, previsoes, probabilidades):
    """Métricas ponderadas (average='weighted'), como parece ser no artigo."""
    return {
        "Precisão": precision_score(y_real, previsoes, average="weighted"),
        "Recall": recall_score(y_real, previsoes, average="weighted"),
        "F1": f1_score(y_real, previsoes, average="weighted"),
        "Acurácia": accuracy_score(y_real, previsoes),
        "AUC": roc_auc_score(y_real, probabilidades),
    }




def parte1_baseline():
    """Parâmetros fixos, treino e teste prontos de cada cenário."""
    linhas = []
    for cenario in CENARIOS:
        X_tr, y_tr = carregar(PASTA / cenario / "train.csv")
        X_te, y_te = carregar(PASTA / cenario / "test.csv")
        modelo = LogisticRegression(**PARAMETROS_BASE).fit(X_tr, y_tr)
        m = metricas(y_te, modelo.predict(X_te), modelo.predict_proba(X_te)[:, 1])
        linhas.append({"Cenário": cenario, **m})
    return pd.DataFrame(linhas)




def parte2_ajustada():
    """Busca de parâmetros por validação cruzada, só no treino.


    A reamostragem (SMOTE/ADASYN) fica DENTRO do pipeline, então em cada fold
    ela só vê o treino daquele fold. Por isso a busca parte de none/train.csv.
    """
    X_tr, y_tr = carregar(PASTA / "none" / "train.csv")
    X_te, y_te = carregar(PASTA / "none" / "test.csv")


    amostradores = {
        "none": "passthrough",
        "smote": SMOTE(random_state=0),
        "adasyn": ADASYN(random_state=0),
    }
    grade = {
        "modelo__C": [0.01, 0.1, 1, 10, 100],
        "modelo__penalty": ["l1", "l2"],
        "modelo__class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    linhas = []
    for cenario, amostrador in amostradores.items():
        pipe = Pipeline(
            [
                ("amostragem", amostrador),
                ("modelo", LogisticRegression(solver="liblinear", max_iter=5000)),
            ]
        )
        busca = GridSearchCV(pipe, grade, scoring="roc_auc", cv=cv, n_jobs=1)
        busca.fit(X_tr, y_tr)


        melhor = busca.best_estimator_
        m = metricas(y_te, melhor.predict(X_te), melhor.predict_proba(X_te)[:, 1])
        parametros = {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()}
        linhas.append(
            {"Cenário": cenario, **m, "AUC (CV)": busca.best_score_, "Parâmetros": parametros}
        )
    return pd.DataFrame(linhas)


def main():
    formato = lambda v: f"{v:.4f}"

    print("Modelo: Regressão Logística")
    print(f"\n=== Parte 1: baseline {PARAMETROS_BASE} ===")
    print(parte1_baseline().to_string(index=False, float_format=formato))

    resultado = parte2_ajustada()
    print("\n=== Parte 2: parâmetros ajustados por validação cruzada ===")
    print(
        resultado.drop(columns=["Parâmetros"]).to_string(
            index=False, float_format=formato
        )
    )

    print("\nMelhores parâmetros por cenário:")
    for _, linha in resultado.iterrows():
        print(f"  {linha['Cenário']:<7} {linha['Parâmetros']}")


if __name__ == "__main__":
    main()