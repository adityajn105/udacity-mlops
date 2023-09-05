"""
Heroku Api test script
"""
import requests


data = {
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
    }
r = requests.post('https://salary-predictor-j8e5.onrender.com/', json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())