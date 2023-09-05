"""
Preprocess data
"""
import pandas as pd


def clean_dataset(df):
    """
    Clean the dataset doing some stuff got from eda
    """
    df.replace({"?": None}, inplace=True)
    df.dropna(inplace=True)

    df.drop("capital-gain", axis="columns", inplace=True)
    df.drop("capital-loss", axis="columns", inplace=True)
    df.drop("fnlgt", axis="columns", inplace=True)
    df.drop("education-num", axis="columns", inplace=True)

    return df


def clean_and_save(
        path="data/census.csv",
        cleaned_path="data/cleaned_census.csv"):
    "Clean and save data"
    df = pd.read_csv(path, skipinitialspace=True)
    df = clean_dataset(df)
    df.to_csv(cleaned_path, index=False)
