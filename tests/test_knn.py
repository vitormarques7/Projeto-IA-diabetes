"""Verificações do protocolo KNN com uma grade pequena e pacientes sintéticos."""

import copy
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import knn
from src.utils.paths import PROJECT_ROOT


GRID = {"model__n_neighbors": [3, 5], "model__weights": ["uniform"], "model__p": [1]}
REFERENCE = {"n_neighbors": 3, "weights": "distance", "metric": "manhattan"}


@pytest.fixture
def trained(split):
    return knn.train_models(split.X_train, split.y_train, GRID, REFERENCE)


def test_import_does_not_load_data_or_train():
    script = '''
import src.data

def forbidden(*args, **kwargs):
    raise AssertionError("Import não deve carregar dados")

src.data.load_split = forbidden
import src.models.knn
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True)
    assert result.stdout == ""


def test_scenarios_metrics_and_training_ids(split, trained):
    fitted, searches, scores, winner = trained
    table = knn.evaluate_models(fitted, scores, split.X_test, split.y_test)
    assert len(table) == 6
    assert np.isfinite(table.select_dtypes(include="number")).all().all()
    assert winner == max(knn.SAMPLING_METHODS, key=lambda s: searches[s].best_score_)
    for key, model in fitted.items():
        clinical = model.named_steps["clinical"]
        assert set(clinical.fit_patient_ids_) == set(split.X_train.index)
        assert set(clinical.fit_patient_ids_).isdisjoint(split.X_test.index)
        assert set(clinical.regression_patient_ids_).issubset(split.X_train.index)
        assert model.named_steps["model"].n_features_in_ == 7
        pred = model.predict(split.X_test)
        prob = model.predict_proba(split.X_test)[:, 1]
        y = split.y_test.to_numpy()
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        comparisons = prob[y == 1, None] - prob[y == 0]
        expected_auc = ((comparisons > 0) + 0.5 * (comparisons == 0)).mean()
        row = table[(table.cenario == key[0]) & (table.modelo == key[1])].iloc[0]
        assert row.teste_accuracy == pytest.approx((pred == y).mean())
        assert row.teste_f1 == pytest.approx(2 * tp / (2 * tp + fp + fn))
        assert row.teste_auc == pytest.approx(expected_auc)


def test_test_data_cannot_affect_selection(split, monkeypatch, tmp_path):
    original_train = knn.train_models
    selections = []
    def small_search(X, y):
        result = original_train(X, y, GRID, REFERENCE)
        selections.append((result[3], {s: x.best_params_ for s, x in result[1].items()}, result[2]))
        return result
    monkeypatch.setattr(knn, "train_models", small_search)
    monkeypatch.setattr(knn, "save_results", lambda *args: None)
    monkeypatch.setattr(knn, "load_split", lambda *args: split)
    knn.run(results_dir=tmp_path)
    changed = copy.deepcopy(split)
    changed.X_test.loc[:, "Glucose"] = 1e8
    changed.y_test = 1 - changed.y_test
    monkeypatch.setattr(knn, "load_split", lambda *args: changed)
    knn.run(results_dir=tmp_path)
    assert selections[0] == selections[1]


def test_exports_and_reload_preserve_predictions(split, trained, tmp_path):
    fitted, searches, scores, winner = trained
    table = knn.evaluate_models(fitted, scores, split.X_test, split.y_test)
    knn.save_results(tmp_path, split, fitted, searches, scores, winner, table)
    saved = pd.read_csv(tmp_path / "metrics/knn_results.csv")
    pd.testing.assert_frame_equal(saved, table, check_exact=False)
    metadata = json.loads((tmp_path / "metrics/knn_experiment.json").read_text())
    assert metadata["grade"] == GRID
    assert metadata["referencia"]["parametros"]["n_neighbors"] == 3
    assert metadata["cenario_selecionado"] == winner
    assert metadata["ids_teste"] == split.X_test.index.tolist()
    for (scenario, kind), model in fitted.items():
        restored = joblib.load(tmp_path / f"artifacts/knn_{scenario}_{kind}.joblib")
        np.testing.assert_array_equal(restored.predict_proba(split.X_test), model.predict_proba(split.X_test))
        assert (tmp_path / f"metrics/knn_cv_{scenario}.csv").is_file()
    best = joblib.load(tmp_path / "artifacts/knn_best.joblib")
    np.testing.assert_array_equal(best.predict(split.X_test), fitted[(winner, "ajustado")].predict(split.X_test))


def test_grid_failures_are_not_silenced(split):
    with pytest.raises(ValueError):
        knn.train_models(split.X_train, split.y_train, {"model__n_neighbors": [10000]}, REFERENCE)
