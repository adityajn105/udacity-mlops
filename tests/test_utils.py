"""
Tests for functions defined in utils
"""

import pandas as pd
from joblib import load
import pytest
import src.utils as utils


@pytest.fixture
def data():
    "Get Dataset"
    df = pd.read_csv("data/cleaned_census.csv")
    return df


def test_processed_datasize(data):
    """Check if processed X and y have same rows"""
    encoder = load("model/encoder.joblib")
    lb = load("model/label_binarizer.joblib")

    X_test, y_test = utils.process_test_data(data, "salary", encoder, lb)
    assert len(X_test) == len(y_test)


def test_encoder_params(data):
    """
    Check if encoder always have same params on same data
    """
    encoder_ref = load("model/encoder.joblib")
    lb_ref = load("model/label_binarizer.joblib")

    _, _, encoder, lb = utils.process_train_data(data, "salary")

    assert encoder_ref.get_params() == encoder.get_params()
    assert lb_ref.get_params() == lb.get_params()


def test_inference_greater():
    """
    Check inference for salary >50k
    """
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/label_binarizer.joblib")

    sample = pd.DataFrame(
        {
            "age": [40],
            "workclass": ["Self-emp-inc"],
            "education": ["Masters"],
            "marital-status": ["Married-AF-spouse"],
            "occupation": ["Exec-managerial"],
            "relationship": ["Husband"],
            "race": ["White"],
            "sex": ["Male"],
            "hours-per-week": [40],
            "native-country": ["England"],
        }
    )

    X = utils.process_inference_data(sample, encoder)
    pred = utils.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"


def test_inference_lower():
    """
    Check inference for salary <50k
    """
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/label_binarizer.joblib")

    sample = pd.DataFrame(
        {
            "age": [60],
            "workclass": ["Self-emp-not-inc"],
            "education": ["9th"],
            "marital-status": ["Never-married"],
            "occupation": ["Farming-fishing"],
            "relationship": ["Husband"],
            "race": ["Black"],
            "sex": ["Male"],
            "hours-per-week": [16],
            "native-country": ["Guatemala"],
        }
    )

    X = utils.process_inference_data(sample, encoder)
    pred = utils.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == "<=50K"
