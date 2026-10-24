"""Gera os três cenários e os registros necessários para reproduzi-los."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.data.download import sources
from src.data.loader import load_split
from src.data.schema import FEATURES, PREDICTORS, SAMPLING_METHODS, SEED, TEST_SIZE, ZERO_MISSING
from src.preprocessing.diagnostics import insulin_rmse, mutual_information
from src.preprocessing.pipeline import ClinicalPreprocessor, make_sampler
from src.utils.paths import PROCESSED_DIR, RAW_DIR, RESULTS_DIR


def prepare_scenario(split, sampling):
    """Devolve tabelas exportáveis, pipeline ajustada e contagens de auditoria."""
    clinical = ClinicalPreprocessor()
    train = clinical.fit_transform(split.X_train, split.y_train)
    test = clinical.transform(split.X_test)
    y_train = split.y_train.copy()
    sampler = make_sampler(sampling)
    if sampling != "none":
        train, y_train = sampler.fit_resample(train, y_train)
    scaler = MinMaxScaler().set_output(transform="pandas")
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    train_out = train_scaled.reset_index(drop=True)
    test_out = test_scaled.reset_index(drop=True)
    train_out["Outcome"] = y_train.to_numpy()
    test_out["Outcome"] = split.y_test.to_numpy()
    preprocessing = Pipeline([
        ("clinical", clinical), ("sampling", sampler), ("scale", scaler)
    ])
    audit = {
        "train_rows_before": len(split.X_train),
        "train_rows_after": len(train_out),
        "test_rows": len(test_out),
        "train_class_counts_before": split.y_train.value_counts().to_dict(),
        "train_class_counts_after": y_train.value_counts().to_dict(),
        "test_class_counts": split.y_test.value_counts().to_dict(),
        "predictor_means": dict(zip(PREDICTORS, clinical.predictor_imputer_.statistics_.tolist())),
        "pima_insulin_mean": clinical.insulin_mean_,
        "insulin_regression_training_ids": clinical.regression_patient_ids_,
        "scaler_data_min": scaler.data_min_.tolist(),
        "scaler_data_max": scaler.data_max_.tolist(),
    }
    return train_out, test_out, preprocessing, audit


def prepare(raw_dir=RAW_DIR, output_dir=PROCESSED_DIR, results_dir=RESULTS_DIR):
    output_dir, results_dir = Path(output_dir), Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in ["metrics", "figures", "artifacts"]:
        (results_dir / folder).mkdir(parents=True, exist_ok=True)
    split = load_split(raw_dir)
    # Divisão bruta permanece disponível para auditoria, nunca para fit global.
    for name in ["train", "test"]:
        X, y = getattr(split, f"X_{name}"), getattr(split, f"y_{name}")
        X.assign(Outcome=y).to_csv(output_dir / f"raw_{name}.csv", index=True)
    split.metadata.to_csv(output_dir / "split.csv", index=True)
    report = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "features": FEATURES,
        "sources": sources(),
        "protocol": {
            "order": ["clinical_imputation", "sampling", "minmax"],
            "zero_as_missing": ZERO_MISSING,
            "insulin_predictors": PREDICTORS,
            "insulin_xgb": {
                "max_depth": 10, "learning_rate": 0.1, "n_estimators": 100,
                "objective": "reg:squarederror", "random_state": SEED, "n_jobs": 1,
            },
            "sampler": {"neighbors": 5, "sampling_strategy": "minority", "random_state": SEED},
            "scaler": {"feature_range": [0, 1], "clip": False},
            "pedigree": "excluded by paper protocol; MI is diagnostic only",
        },
        "python": platform.python_version(),
        "versions": {pkg: version(pkg) for pkg in [
            "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "imbalanced-learn",
            "openpyxl", "joblib", "matplotlib", "pyparsing",
        ]},
        "dataset_rows": len(split.metadata),
        "class_counts": pd.concat([split.y_train, split.y_test]).value_counts().to_dict(),
        "insulin_regression": insulin_rmse(split.X_train),
        "scenarios": {},
    }
    mi = mutual_information(split.X_train, split.y_train)
    mi.to_csv(results_dir / "metrics" / "mutual_information.csv")
    # Cache local: notebooks e execução restrita não precisam escrever no HOME.
    cache = (results_dir / ".cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    mi.plot.bar(ax=ax, color="#247a8a")
    ax.set_ylabel("Mutual information")
    ax.set_title("Pima: apenas pacientes do treino externo")
    fig.savefig(results_dir / "figures" / "mutual_information.png", dpi=150)
    plt.close(fig)
    for sampling in SAMPLING_METHODS:
        train, test, pipeline, audit = prepare_scenario(split, sampling)
        target = output_dir / sampling
        target.mkdir(parents=True, exist_ok=True)
        train.to_csv(target / "train.csv", index=False)
        test.to_csv(target / "test.csv", index=False)
        joblib.dump(pipeline, results_dir / "artifacts" / f"preprocessing_{sampling}.joblib")
        report["scenarios"][sampling] = audit
        print(f"{sampling}: treino {len(train)}, teste {len(test)}; {target}")
    report_path = results_dir / "metrics" / "preprocessing.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if not report["insulin_regression"]["xgb_is_best_local"]:
        print("XGB mantido por protocolo; outro regressor teve menor RMSE local. Consulte o relatório.")
    print(f"Relatório: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    prepare(args.raw_dir, args.output_dir, args.results_dir)


if __name__ == "__main__":
    main()
