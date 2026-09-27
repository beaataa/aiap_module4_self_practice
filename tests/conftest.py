"""shared fixtures. puts src/ on the path so the modules import as in run.sh."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_preparation import load_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture
def raw_rows():
    """a handful of rows covering every messy case the cleaning has to handle.

    written by hand rather than sampled, so each row is here for a reason:
    both target styles, both smoke styles, both ejection fraction styles, a
    negative age, a missing creatinine, and Pre-diabetes.
    """
    return pd.DataFrame([
        # numeric target, lowercase smoke, spelled-out EF, normal age
        {"ID": "AAA111", "Survive": "0", "Gender": "Male", "Smoke": "Yes",
         "Diabetes": "Normal", "Age": 60, "Ejection Fraction": "Low",
         "Sodium": 140, "Creatinine": 1.0, "Platelets": 250000.0,
         "Creatine phosphokinase": 100, "Blood Pressure": 80, "Hemoglobin": 14.0,
         "Height": 170, "Weight": 70, "Favorite color": "red"},
        # text target, uppercase smoke, abbreviated EF, negative age
        {"ID": "BBB222", "Survive": "Yes", "Gender": "Female", "Smoke": "NO",
         "Diabetes": "Pre-diabetes", "Age": -55, "Ejection Fraction": "N",
         "Sodium": 135, "Creatinine": 2.0, "Platelets": 200000.0,
         "Creatine phosphokinase": 200, "Blood Pressure": 150, "Hemoglobin": 10.0,
         "Height": 160, "Weight": 90, "Favorite color": "blue"},
        # text target "No", abbreviated EF "L", missing creatinine
        {"ID": "CCC333", "Survive": "No", "Gender": "Male", "Smoke": "YES",
         "Diabetes": "Diabetes", "Age": 70, "Ejection Fraction": "L",
         "Sodium": 130, "Creatinine": None, "Platelets": 300000.0,
         "Creatine phosphokinase": 300, "Blood Pressure": 120, "Hemoglobin": 11.9,
         "Height": 150, "Weight": 45, "Favorite color": "green"},
        # numeric target "1", exactly on the anaemia boundary
        {"ID": "DDD444", "Survive": "1", "Gender": "Female", "Smoke": "No",
         "Diabetes": "Normal", "Age": 45, "Ejection Fraction": "High",
         "Sodium": 145, "Creatinine": 0.7, "Platelets": 275000.0,
         "Creatine phosphokinase": 50, "Blood Pressure": 119, "Hemoglobin": 12.0,
         "Height": 180, "Weight": 81, "Favorite color": "black"},
    ])


@pytest.fixture(scope="session")
def cleaned(config):
    """the real cleaned dataset. session-scoped, it is the slow fixture."""
    from data_preparation import prepare_data
    return prepare_data(config)
