"""
Render Api test script
"""
import requests


data = {
    "age": 40,
    "workclass": "Self-emp-inc",
    "education": "Masters",
    "marital_status": "Married-AF-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "hours_per_week": 40,
    "native_country": "England",
    }
r = requests.post('https://salary-predictor-j8e5.onrender.com/', json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
