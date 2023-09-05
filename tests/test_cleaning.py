"Test Cleaning"

import pandas as pd
import pytest
import src.cleaning


@pytest.fixture
def data():
    "Dataset"
    df = pd.read_csv("data/census.csv", skipinitialspace=True)
    df = src.cleaning.clean_dataset(df)
    return df


def test_if_question_mark_present(data):
    "Data should not have question marks"
    assert "?" not in data.values


def test_if_null_present(data):
    "Data should not have null values"
    assert data.shape == data.dropna().shape


def test_if_removed_columns_present(data):
    "Data should not have dropped columns"
    for col in ["fnlgt", "capital-gain", "capital-loss", "education-num"]:
        assert col not in data.columns
