"""
Train model
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
import src.utils as utils


def train_and_save(path="data/cleaned_census.csv"):
    """
    Split data, train model and save files
    """
    df = pd.read_csv(path)
    train, _ = train_test_split(df, test_size=0.2)

    X_train, y_train, encoder, lb = utils.process_train_data(
        train, "salary")
    model = utils.train_model(X_train, y_train)

    dump(model, "model/model.joblib")
    dump(encoder, "model/encoder.joblib")
    dump(lb, "model/label_binarizer.joblib")
