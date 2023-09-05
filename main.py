# Put the code for your API here.

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import src.utils as utils
import pandas as pd
from joblib import load


class User(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    marital_status: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
        'Widowed']
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
        'Craft-repair', 'Protective-serv', 'Armed-Forces',
        'Priv-house-serv']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child',
        'Unmarried', 'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
        'Other']
    sex: Literal['Male', 'Female']
    hours_per_week: int
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']


app = FastAPI()


@app.get("/")
async def home():
    return "Welcome to Salary Predictor."


@app.post("/")
async def inference(user_data: User):
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/label_binarizer.joblib")

    data = pd.DataFrame(
        {
            "age": [user_data.age],
            "workclass": [user_data.workclass],
            "education": [user_data.education],
            "marital-status": [user_data.marital_status],
            "occupation": [user_data.occupation],
            "relationship": [user_data.occupation],
            "race": [user_data.race],
            "sex": [user_data.sex],
            "hours-per-week": [user_data.hours_per_week],
            "native-country": [user_data.native_country],
        }
    )

    X = utils.process_inference_data(data, encoder)
    pred = utils.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    return {"prediction": y}
