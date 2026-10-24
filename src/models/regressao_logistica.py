"""Regressão Logística para predição de diabetes (cenários none, smote e adasyn)."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# parents[2] sobe duas pastas (models -> src -> raiz do projeto).
RAIZ = Path(__file__).resolve().parents[2]

# Pasta com os CSVs gerados pelo pré-processamento (um subdiretório por cenário).
PASTA_PROCESSADOS = RAIZ / "data" / "processed"

# Os três cenários de balanceamento que o guia pede.
CENARIOS = ["none", "smote", "adasyn"]

# Coluna que diz se a pessoa tem diabetes (0 = não, 1 = sim).
COLUNA_ALVO = "Outcome"

# Parâmetros fixos do modelo (iguais nos três cenários, para a comparação ser justa).
PARAMETROS = {"max_iter": 1000, "random_state": 42}


def carregar(caminho):
    """Lê um CSV e separa em X (características) e y (resposta correta)."""
    df = pd.read_csv(caminho)

    if COLUNA_ALVO not in df.columns:
        raise ValueError(
            f"Coluna '{COLUNA_ALVO}' não encontrada. Colunas: {list(df.columns)}"
        )

    return df.drop(columns=[COLUNA_ALVO]), df[COLUNA_ALVO]


def avaliar_cenario(cenario):
    """Treina e avalia a regressão logística em um cenário. Devolve as métricas."""
    # Os CSVs já vêm divididos e normalizados: não refazer nada disso.
    X_treino, y_treino = carregar(PASTA_PROCESSADOS / cenario / "train.csv")
    X_teste, y_teste = carregar(PASTA_PROCESSADOS / cenario / "test.csv")

    modelo = LogisticRegression(**PARAMETROS)
    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)
    probabilidades = modelo.predict_proba(X_teste)[:, 1]

    return {
        "Cenário": cenario,
        "Treino": len(X_treino),
        "Teste": len(X_teste),
        "Acurácia": accuracy_score(y_teste, previsoes),
        "F1-Score": f1_score(y_teste, previsoes),
        "AUC": roc_auc_score(y_teste, probabilidades),
    }


def main():
    resultados = [avaliar_cenario(c) for c in CENARIOS]

    tabela = pd.DataFrame(resultados)
    print("Modelo: Regressão Logística")
    print(f"Parâmetros: {PARAMETROS}\n")
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()