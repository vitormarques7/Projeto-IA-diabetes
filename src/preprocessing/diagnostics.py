"""Diagnósticos restritos ao treino externo; não selecionam características."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

from src.data.schema import PIMA_COLUMNS, PREDICTORS, SEED, ZERO_MISSING
from src.preprocessing.pipeline import make_insulin_regressor


def insulin_rmse(X_train):
    """Compara modelos no Pima com insulina realmente observada.

    SVR e GPR usam os defaults das versões fixadas. Todas as entradas mantêm
    unidades originais nesta comparação; não há ajuste por dados de teste.
    """
    frame = X_train.loc[X_train["_source"].eq("pima")].copy()
    frame[ZERO_MISSING] = frame[ZERO_MISSING].replace(0, np.nan)
    frame = frame.loc[frame["Insulin"].notna()]
    if len(frame) < 10:
        raise ValueError("Poucas insulinas observadas para a comparação interna")
    inner_train, inner_valid = train_test_split(frame, test_size=0.20, random_state=SEED)
    regressors = {
        "XGB": make_insulin_regressor(),
        "SVR": SVR(),
        "GPR": GaussianProcessRegressor(random_state=SEED),
    }
    scores = {}
    for name, regressor in regressors.items():
        pipeline = make_pipeline(SimpleImputer(strategy="mean"), regressor)
        pipeline.fit(inner_train[PREDICTORS], inner_train["Insulin"])
        predicted = pipeline.predict(inner_valid[PREDICTORS])
        scores[name] = float(np.sqrt(mean_squared_error(inner_valid["Insulin"], predicted)))
    return {
        "rmse": scores,
        "best_local": min(scores, key=scores.get),
        "selected_by_protocol": "XGB",
        "xgb_is_best_local": min(scores, key=scores.get) == "XGB",
        "train_patient_ids": inner_train.index.tolist(),
        "validation_patient_ids": inner_valid.index.tolist(),
        "predictors": PREDICTORS,
        "comparison_preprocessing": "mean imputation; original units; no target imputation",
        "svr_gpr_parameters": "defaults of pinned versions; GPR random_state=0",
    }


def mutual_information(X_train, y_train):
    pima = X_train.loc[X_train["_source"].eq("pima")].copy()
    features = [c for c in PIMA_COLUMNS if c != "Outcome"]
    values = pima[features].copy()
    values[ZERO_MISSING] = values[ZERO_MISSING].replace(0, np.nan)
    imputed = SimpleImputer(strategy="mean").fit_transform(values)
    scores = mutual_info_classif(imputed, y_train.loc[pima.index], random_state=SEED)
    return pd.Series(scores, index=features, name="mutual_information").sort_values(ascending=False)
