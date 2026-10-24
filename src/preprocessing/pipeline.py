"""Transformações ajustáveis por fold, sem usar rótulos para gerar insulina."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor

from src.data.schema import FEATURES, PREDICTORS, SAMPLING_METHODS, SEED, ZERO_MISSING


def make_insulin_regressor():
    return XGBRegressor(
        max_depth=10, learning_rate=0.1, n_estimators=100,
        objective="reg:squarederror", random_state=SEED, n_jobs=1,
    )


def clinical_values(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Use um DataFrame com nomes de características e _source")
    required = FEATURES + ["_source"]
    missing = set(required) - set(X.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    if "Outcome" in X.columns:
        raise ValueError("Outcome deve ser fornecido separadamente como y")
    if not X["_source"].isin(["pima", "rtml"]).all():
        raise ValueError("Origem deve ser pima ou rtml")
    values = X[FEATURES].apply(pd.to_numeric, errors="raise").copy()
    values[ZERO_MISSING] = values[ZERO_MISSING].replace(0, np.nan)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Características contêm infinito")
    return values


class ClinicalPreprocessor(TransformerMixin, BaseEstimator):
    """Aprende médias e regressão somente dos pacientes recebidos em fit.

    _source escolhe a regra de imputação e é descartado na saída. Outcome não
    participa da regressão; DiabetesPedigreeFunction é excluída por protocolo.
    """

    def fit(self, X, y=None):
        values = clinical_values(X)
        if values[PREDICTORS].isna().all().any():
            raise ValueError("Há preditor inteiramente ausente no treino")
        self.predictor_imputer_ = SimpleImputer(strategy="mean").fit(values[PREDICTORS])
        observed = X["_source"].eq("pima") & values["Insulin"].notna()
        if observed.sum() < 2:
            raise ValueError("Treino requer ao menos duas insulinas Pima observadas")
        self.insulin_mean_ = float(values.loc[observed, "Insulin"].mean())
        self.regression_patient_ids_ = X.index[observed].tolist()
        self.fit_patient_ids_ = X.index.tolist()
        self.insulin_regressor_ = make_insulin_regressor()
        predictors = self.predictor_imputer_.transform(values.loc[observed, PREDICTORS])
        self.insulin_regressor_.fit(predictors, values.loc[observed, "Insulin"])
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "insulin_regressor_")
        values = clinical_values(X)
        values[PREDICTORS] = self.predictor_imputer_.transform(values[PREDICTORS])
        missing = values["Insulin"].isna()
        rtml_missing = missing & X["_source"].eq("rtml")
        if rtml_missing.any():
            values.loc[rtml_missing, "Insulin"] = self.insulin_regressor_.predict(
                values.loc[rtml_missing, PREDICTORS].to_numpy()
            )
        values.loc[missing & ~rtml_missing, "Insulin"] = self.insulin_mean_
        result = values[FEATURES]
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Pré-processamento produziu valores não finitos")
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)


def make_sampler(sampling):
    if sampling not in SAMPLING_METHODS:
        raise ValueError(f"sampling deve ser um de {SAMPLING_METHODS}")
    if sampling == "none":
        return "passthrough"
    if sampling == "smote":
        return SMOTE(sampling_strategy="minority", k_neighbors=5, random_state=SEED)
    return ADASYN(sampling_strategy="minority", n_neighbors=5, random_state=SEED)


def build_pipeline(estimator, sampling="none"):
    """Use com GridSearchCV sobre X/y brutos de load_split(), nunca CSVs prontos.

    Ordem dos autores: limpeza/insulina -> reamostragem -> Min-Max -> modelo.
    A amostragem só acontece em fit; predict/score preservam o teste.
    """
    if estimator is None:
        raise ValueError("Forneça o classificador do integrante")
    return Pipeline([
        ("clinical", ClinicalPreprocessor()),
        ("sampling", make_sampler(sampling)),
        ("scale", MinMaxScaler().set_output(transform="pandas")),
        ("model", estimator),
    ])
