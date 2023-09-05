"""
Tests for fastapi server
"""
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """
    Get client
    """
    return TestClient(app)


def test_home(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to Salary Predictor."


def test_invalid_url(client):
    r = client.get("/invalid")
    assert r.status_code != 200


def test_salary_greater(client):
    r = client.post(
        "/",
        json={
            "age": 40,
            "workclass": "Private",
            "education": "Doctorate",
            "marital_status": "Never-married",
            "occupation": "Exec-managerial",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "hours_per_week": 40,
            "native_country": "United-States",
        },
    )
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_post_below(client):
    r = client.post(
        "/",
        json={
            "age": 60,
            "workclass": "Self-emp-not-inc",
            "education": "9th",
            "marital_status": "Never-married",
            "occupation": "Farming-fishing",
            "relationship": "Husband",
            "race": "Black",
            "sex": "Male",
            "hours_per_week": 16,
            "native_country": "Guatemala",
        },
    )
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}
