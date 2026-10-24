"""Regressão Logística para predição de diabetes."""

# Path: forma moderna de trabalhar com caminhos de arquivos/pastas.
from pathlib import Path

# pandas: biblioteca para ler e manipular tabelas (como o Excel, mas em código).
import pandas as pd

# LogisticRegression: o modelo de regressão logística, pronto no scikit-learn.
from sklearn.linear_model import LogisticRegression

# Funções que calculam as métricas usadas no artigo para comparar resultados.
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# __file__ é o caminho deste arquivo (src/models/regressao_logistica.py).
# parents[2] sobe duas pastas (models -> src -> raiz do projeto).
# Assim o código funciona em qualquer computador, sem depender de C:\Users\...
RAIZ = Path(__file__).resolve().parents[2]

# Os CSVs já tratados pelos colegas, que ficam na raiz do projeto.
ARQUIVO_TREINO = RAIZ / "diabetes_treino_processado.csv"
ARQUIVO_TESTE = RAIZ / "diabetes_teste_processado.csv"

# Nome da coluna que diz se a pessoa tem diabetes (0 = não, 1 = sim).
# É o que o modelo tenta prever. Se o nome for outro no CSV, troque aqui.
COLUNA_ALVO = "Outcome"


def carregar(caminho):
    """Lê um CSV e separa em X (características) e y (resposta correta)."""
    # Lê o arquivo CSV e transforma em uma tabela (DataFrame).
    df = pd.read_csv(caminho)

    # Proteção: se a coluna alvo não existir, avisa com uma mensagem clara
    # e mostra os nomes reais das colunas, em vez de dar um erro confuso.
    if COLUNA_ALVO not in df.columns:
        raise ValueError(
            f"Coluna '{COLUNA_ALVO}' não encontrada. Colunas: {list(df.columns)}"
        )

    # X = todas as colunas menos a alvo (idade, glicose, IMC, etc.).
    # y = só a coluna alvo (a resposta que o modelo deve aprender a prever).
    return df.drop(columns=[COLUNA_ALVO]), df[COLUNA_ALVO]


def main():
    # Carrega os dois conjuntos:
    # treino = dados que o modelo usa para APRENDER;
    # teste  = dados que ele nunca viu, usados para AVALIAR se aprendeu bem.
    X_treino, y_treino = carregar(ARQUIVO_TREINO)
    X_teste, y_teste = carregar(ARQUIVO_TESTE)

    # Cria o modelo.
    # max_iter=1000: número máximo de tentativas do algoritmo para ajustar o
    #   modelo (o padrão, 100, às vezes não basta e gera um aviso).
    # random_state=42: fixa a "sorte" do algoritmo, para o resultado ser
    #   sempre o mesmo toda vez que rodar (reprodutível).
    modelo = LogisticRegression(max_iter=1000, random_state=42)

    # fit = treinar. O modelo olha para X_treino e y_treino e aprende a
    # relação entre as características e ter ou não diabetes.
    modelo.fit(X_treino, y_treino)

    # predict = o modelo responde 0 ou 1 para cada paciente do teste.
    previsoes = modelo.predict(X_teste)

    # predict_proba = em vez de 0 ou 1, devolve a PROBABILIDADE de cada classe.
    # [:, 1] pega só a coluna da probabilidade de ter diabetes (classe 1).
    # O AUC precisa dessas probabilidades, não só do 0/1.
    probabilidades = modelo.predict_proba(X_teste)[:, 1]

    # Compara o que o modelo previu com a resposta correta (y_teste):
    # Acurácia: porcentagem de acertos no total.
    print(f"Acurácia: {accuracy_score(y_teste, previsoes):.4f}")
    # F1-Score: equilíbrio entre precisão e recall; útil quando as classes
    #   são desiguais (aqui há menos pessoas com diabetes do que sem).
    print(f"F1-Score: {f1_score(y_teste, previsoes):.4f}")
    # AUC: mede o quanto o modelo separa bem quem tem de quem não tem
    #   diabetes (1.0 = perfeito, 0.5 = chute aleatório).
    print(f"AUC:      {roc_auc_score(y_teste, probabilidades):.4f}")


# Só executa main() quando você roda este arquivo diretamente
# (python src/models/regressao_logistica.py). Se outro arquivo importar este,
# nada roda sozinho.
if __name__ == "__main__":
    main()