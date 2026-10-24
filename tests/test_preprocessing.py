import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data.download import sources, verify_bytes, download
from src.data.loader import load_split
from src.data.schema import FEATURES, PREDICTORS
from src.preprocessing.diagnostics import insulin_rmse, mutual_information
from src.preprocessing.pipeline import ClinicalPreprocessor, build_pipeline
from src.preprocessing.prepare import prepare_scenario, prepare


def test_published_sources_and_split(real_split):
    s = real_split
    assert len(s.metadata) == 877
    assert (len(s.X_train), len(s.X_test)) == (701, 176)
    assert s.y_train.value_counts().to_dict() == {0: 468, 1: 233}
    assert s.y_test.value_counts().to_dict() == {0: 118, 1: 58}
    assert s.metadata.source.value_counts().to_dict() == {"pima": 768, "rtml": 109}
    assert not set(s.X_train.index) & set(s.X_test.index)
    assert "Outcome" not in s.X_train
    repeated = load_split()
    pd.testing.assert_frame_equal(s.X_train, repeated.X_train)
    assert s.X_train["Glucose"].eq(0).any()  # Bruto ainda não imputado.
    assert s.X_train.loc[s.X_train._source.eq("rtml"), "Insulin"].isna().all()


def test_download_checksums_and_no_overwrite(tmp_path, monkeypatch):
    spec = sources()["pima"]
    target = tmp_path / spec["filename"]
    target.write_bytes(b"different local data")
    def unexpected_network(*args, **kwargs):
        pytest.fail("Arquivo existente deve ser validado antes de acessar a rede")
    monkeypatch.setattr("src.data.download.urlopen", unexpected_network)
    with pytest.raises(ValueError, match="Checksum"):
        download(tmp_path)
    assert target.read_bytes() == b"different local data"
    with pytest.raises(ValueError, match="Checksum"):
        verify_bytes(b"bad", spec)


def test_observed_values_zero_rules_and_feature_contract(split):
    processor = ClinicalPreprocessor().fit(split.X_train, split.y_train)
    actual = processor.transform(split.X_test)
    assert actual.columns.tolist() == FEATURES
    assert np.isfinite(actual.to_numpy()).all()
    for col in FEATURES:
        expected = split.X_test[col]
        observed = expected.notna()
        if col not in ["Pregnancies", "Age"]:
            observed &= expected.ne(0)
        np.testing.assert_array_equal(actual.loc[observed, col], expected.loc[observed])
    np.testing.assert_array_equal(actual.Pregnancies, split.X_test.Pregnancies)
    known = split.X_train._source.eq("pima") & split.X_train.Insulin.gt(0)
    assert processor.insulin_mean_ == split.X_train.loc[known, "Insulin"].mean()
    assert set(processor.regression_patient_ids_) == set(split.X_train.index[known])
    assert processor.insulin_regressor_.n_features_in_ == 6


def test_labels_metadata_and_pedigree_cannot_influence_imputation(split):
    a = ClinicalPreprocessor().fit(split.X_train, split.y_train)
    changed = split.X_train.copy()
    changed["DiabetesPedigreeFunction"] = 999999
    changed.index = [f"other-id:{i}" for i in range(len(changed))]
    b = ClinicalPreprocessor().fit(changed, 1 - split.y_train.to_numpy())
    pd.testing.assert_frame_equal(a.transform(split.X_test), b.transform(split.X_test))
    with pytest.raises(ValueError, match="Outcome"):
        a.transform(split.X_test.assign(Outcome=split.y_test))


def test_transform_cannot_learn_from_test(split):
    model = build_pipeline(LogisticRegression(max_iter=300), "smote")
    model.fit(split.X_train, split.y_train)
    clinical, scaler = model.named_steps["clinical"], model.named_steps["scale"]
    before = copy.deepcopy((clinical.predictor_imputer_.statistics_, scaler.data_min_, scaler.data_max_))
    before_regression = clinical.insulin_regressor_.predict(np.ones((2, 6)))
    extreme = split.X_test.copy()
    extreme[PREDICTORS] = 1e8
    model.predict(extreme)
    after = (clinical.predictor_imputer_.statistics_, scaler.data_min_, scaler.data_max_)
    for expected, actual in zip(before, after):
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(before_regression, clinical.insulin_regressor_.predict(np.ones((2, 6))))
    assert scaler.transform(clinical.transform(extreme)).to_numpy().max() > 1
    assert set(clinical.fit_patient_ids_).isdisjoint(split.X_test.index)
    assert model.named_steps["model"].feature_names_in_.tolist() == FEATURES


def test_test_values_and_labels_cannot_change_training_exports(split):
    before, _, before_pipeline, _ = prepare_scenario(split, "adasyn")
    altered = copy.deepcopy(split)
    altered.X_test[PREDICTORS] = 1e8
    altered.y_test = 1 - split.y_test
    after, altered_test, after_pipeline, _ = prepare_scenario(altered, "adasyn")
    pd.testing.assert_frame_equal(before, after)
    np.testing.assert_array_equal(altered_test.Outcome, altered.y_test)
    for attribute in ["data_min_", "data_max_"]:
        np.testing.assert_array_equal(
            getattr(before_pipeline.named_steps["scale"], attribute),
            getattr(after_pipeline.named_steps["scale"], attribute),
        )


@pytest.mark.parametrize("sampling", ["none", "smote", "adasyn"])
def test_scenarios_preserve_test_and_are_reproducible(split, sampling, tmp_path):
    train, test, pipeline, audit = prepare_scenario(split, sampling)
    repeated_train, repeated_test, _, _ = prepare_scenario(split, sampling)
    pd.testing.assert_frame_equal(train, repeated_train)
    pd.testing.assert_frame_equal(test, repeated_test)
    assert train.columns.tolist() == FEATURES + ["Outcome"]
    assert len(test) == len(split.X_test)
    np.testing.assert_array_equal(test.Outcome, split.y_test)
    assert not train.isna().any().any()
    assert not test.isna().any().any()
    assert len(train) >= len(split.X_train)
    if sampling == "none":
        assert len(train) == len(split.X_train)
    if sampling == "smote":
        counts = train.Outcome.value_counts()
        assert counts[0] == counts[1]
    assert audit["test_class_counts"] == split.y_test.value_counts().to_dict()
    artifact = tmp_path / "preprocessing.joblib"
    joblib.dump(pipeline, artifact)
    restored = joblib.load(artifact)
    pd.testing.assert_frame_equal(
        restored.transform(split.X_test).reset_index(drop=True), test[FEATURES]
    )


def test_cross_validation_refits_all_steps_inside_folds(split):
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    model = build_pipeline(LogisticRegression(max_iter=300), "smote")
    result = cross_validate(
        model, split.X_train, split.y_train, cv=folds,
        scoring="roc_auc", return_estimator=True, error_score="raise",
    )
    assert np.isfinite(result["test_score"]).all()
    for fitted, (train_idx, valid_idx) in zip(result["estimator"], folds.split(split.X_train, split.y_train)):
        train, valid = split.X_train.iloc[train_idx], split.X_train.iloc[valid_idx]
        clinical = fitted.named_steps["clinical"]
        assert set(clinical.fit_patient_ids_) == set(train.index)
        assert set(clinical.regression_patient_ids_).isdisjoint(valid.index)
        assert set(clinical.regression_patient_ids_).issubset(train.index)
        expected = train[PREDICTORS].replace({"Glucose": {0: np.nan}, "BMI": {0: np.nan}}).mean()
        np.testing.assert_allclose(clinical.predictor_imputer_.statistics_, expected)
        assert fitted.named_steps["model"].n_features_in_ == 7


def test_diagnostics_are_restricted_to_outer_training(split):
    result = insulin_rmse(split.X_train)
    ids = set(result["train_patient_ids"]) | set(result["validation_patient_ids"])
    assert ids.issubset(split.X_train.index)
    assert ids.isdisjoint(split.X_test.index)
    assert not set(result["train_patient_ids"]) & set(result["validation_patient_ids"])
    assert split.X_train.loc[list(ids), "Insulin"].gt(0).all()
    assert set(result["rmse"]) == {"XGB", "SVR", "GPR"}
    assert result["predictors"] == PREDICTORS
    mi = mutual_information(split.X_train, split.y_train)
    assert len(mi) == 8 and "DiabetesPedigreeFunction" in mi.index


@pytest.mark.parametrize("problem", ["missing_source", "unknown_source", "infinity", "no_insulin"])
def test_invalid_training_inputs_fail_clearly(split, problem):
    X = split.X_train.copy()
    if problem == "missing_source":
        X = X.drop(columns="_source")
    elif problem == "unknown_source":
        X["_source"] = "unknown"
    elif problem == "infinity":
        X["Glucose"] = np.inf
    elif problem == "no_insulin":
        X["Insulin"] = np.nan
    with pytest.raises(ValueError):
        ClinicalPreprocessor().fit(X)


def test_full_export_has_auditable_outputs(real_split, tmp_path):
    report = prepare(output_dir=tmp_path / "processed", results_dir=tmp_path / "results")
    assert report["dataset_rows"] == 877
    assert set(report["scenarios"]) == {"none", "smote", "adasyn"}
    for name in report["scenarios"]:
        test = pd.read_csv(tmp_path / "processed" / name / "test.csv")
        assert len(test) == 176
        np.testing.assert_array_equal(test.Outcome, real_split.y_test)
        assert test.columns.tolist() == FEATURES + ["Outcome"]
    saved = json.loads((tmp_path / "results" / "metrics" / "preprocessing.json").read_text())
    assert saved["sources"] == sources()
    assert saved["protocol"]["insulin_predictors"] == PREDICTORS
    assert saved["protocol"]["scaler"]["clip"] is False
    assert (tmp_path / "results" / "figures" / "mutual_information.png").is_file()
