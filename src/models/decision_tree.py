
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

cenarios = ["none", "smote", "adasyn"]

for cenario in cenarios:
    print("\nÁRVORE DE DECISÃO -", cenario.upper())

    # Carregar os dados
    df_treino = pd.read_csv(f"data/processed/{cenario}/train.csv")
    df_teste = pd.read_csv(f"data/processed/{cenario}/test.csv")

    # Separar os dados
    X_treino = df_treino.drop(columns="Outcome")
    y_treino = df_treino["Outcome"]

    X_test = df_teste.drop(columns="Outcome")
    y_test = df_teste["Outcome"]

    # Criar o modelo
    dt = tree.DecisionTreeClassifier(
        criterion="gini",
        max_depth=2,
        min_samples_leaf=50,
        random_state=0
    )

    dt.fit(X_treino, y_treino)

    # Accuracy on Train
    print("Training Accuracy is:", dt.score(X_treino, y_treino))

    # Accuracy on Test
    print("Testing Accuracy is:", dt.score(X_test, y_test))

    # Fazer previsões
    X_test_prediction = dt.predict(X_test)

    print(classification_report(y_test, X_test_prediction, zero_division=0))

    # Calcular AUC
    X_test_probability = dt.predict_proba(X_test)[:, 1]
    print("AUC:", roc_auc_score(y_test, X_test_probability))
