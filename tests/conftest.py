import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data.loader import DataSplit, load_split
from src.data.schema import PREDICTORS


@pytest.fixture
def split():
    """Pacientes sintéticos para verificar invariantes, sem acesso à rede."""
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.uniform(10, 100, size=(120, 6)), columns=PREDICTORS)
    frame["Pregnancies"] = rng.integers(0, 9, size=len(frame)).astype(float)
    frame["DiabetesPedigreeFunction"] = rng.uniform(0.1, 1, size=len(frame))
    frame["Insulin"] = 1.5 * frame["Glucose"] + rng.uniform(1, 10, size=len(frame))
    frame["_source"] = ["pima"] * 96 + ["rtml"] * 24
    frame.loc[96:, ["Insulin", "DiabetesPedigreeFunction"]] = np.nan
    frame.loc[::7, "Glucose"] = 0
    frame.loc[::11, "BMI"] = 0
    frame.loc[:10, "Insulin"] = 0
    frame.index = pd.Index([f"synthetic:{i}" for i in range(len(frame))], name="patient_id")
    y = pd.Series(np.tile([0, 0, 1], 40), index=frame.index, name="Outcome")
    X_train, X_test, y_train, y_test = train_test_split(
        frame, y, test_size=0.2, stratify=y, random_state=0
    )
    metadata = frame[["_source"]].rename(columns={"_source": "source"})
    metadata["partition"] = "train"
    metadata.loc[X_test.index, "partition"] = "test"
    return DataSplit(X_train, X_test, y_train, y_test, metadata)


@pytest.fixture(scope="session")
def real_split():
    try:
        return load_split()
    except FileNotFoundError:
        pytest.skip("Integração: execute python -m src.data.download antes dos testes")
