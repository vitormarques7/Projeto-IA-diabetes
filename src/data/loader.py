"""Leitura validada e divisão compartilhada, sem aprender transformações."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.download import sources, verify_file
from src.data.schema import FEATURES, PIMA_COLUMNS, SEED, TEST_SIZE
from src.utils.paths import RAW_DIR


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metadata: pd.DataFrame


def load_datasets(raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    manifest = sources()
    for spec in manifest.values():
        verify_file(raw_dir / spec["filename"], spec)
    pima = pd.read_csv(raw_dir / manifest["pima"]["filename"], header=None, names=PIMA_COLUMNS)
    rtml = pd.read_excel(raw_dir / manifest["rtml"]["filename"], usecols=FEATURES + ["Outcome"])
    for name, frame in [("pima", pima), ("rtml", rtml)]:
        if len(frame) != manifest[name]["rows"]:
            raise ValueError(f"Número inesperado de pacientes em {name}: {len(frame)}")
        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not frame["Outcome"].isin([0, 1]).all():
            raise ValueError(f"Outcome inválido em {name}")
        required = [c for c in frame.columns if not (name == "rtml" and c == "Insulin")]
        if not np.isfinite(frame[required].to_numpy()).all():
            raise ValueError(f"Valor ausente ou infinito inesperado em {name}")
        frame["Outcome"] = frame["Outcome"].astype(int)
        frame["_source"] = name
        frame.index = pd.Index([f"{name}:{i:04d}" for i in range(len(frame))], name="patient_id")
    if not rtml["Insulin"].isna().all():
        raise ValueError("Esperada fonte RTML sem insulina observada")
    merged = pd.concat([pima, rtml], axis=0)
    if merged["Outcome"].value_counts().to_dict() != {0: 586, 1: 291}:
        raise ValueError("Distribuição de classes diferente da versão publicada")
    return merged


def load_split(raw_dir=RAW_DIR):
    """X inclui _source para imputação; a pipeline o remove antes do classificador.

    O índice patient_id acompanha a origem. Nenhum valor é imputado ou escalado
    nesta função. Não altere esta divisão individualmente em cada modelo.
    """
    frame = load_datasets(raw_dir)
    X = frame.drop(columns="Outcome")
    y = frame["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    metadata = frame[["_source"]].rename(columns={"_source": "source"})
    metadata["partition"] = "train"
    metadata.loc[X_test.index, "partition"] = "test"
    return DataSplit(X_train, X_test, y_train, y_test, metadata)
