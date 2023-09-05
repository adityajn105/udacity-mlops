"""
Utilities for process, split, train, inference
"""
import logging
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def process_train_data(X, label):
    """
    Process the data for training

    Input:
            X: Input dataframe
            label: name of label column
    Returns:
            X: processed data
            y: label
            encoder: categorical feature encoder
            label_binarizer: to bianrize label
    """
    y = X[label]
    X.drop(columns=[label], inplace=True)

    X_cat = X[categorical_features].values
    X_cont = X.drop(columns=categorical_features)

    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    lb = LabelBinarizer()
    X_cat = encoder.fit_transform(X_cat)
    y = lb.fit_transform(y.values).ravel()
    X = np.concatenate([X_cat, X_cont], axis=1)
    return X, y, encoder, lb


def process_test_data(X, label, encoder, lb):
    """
    Use fitted encoder and labelbinarizer to transform test dataset
    """
    y = X[label]
    X.drop(columns=[label], inplace=True)
    X_cat = X[categorical_features].values
    X_cont = X.drop(columns=categorical_features)

    X_cat = encoder.transform(X_cat)
    y = lb.transform(y.values).ravel()
    X = np.concatenate([X_cat, X_cont], axis=1)
    return X, y


def process_inference_data(X, encoder):
    """
    Use fitted encoder to transform inference dataset
    """
    X_cat = X[categorical_features].values
    X_cont = X.drop(columns=categorical_features)
    X_cat = encoder.transform(X_cat)
    X = np.concatenate([X_cat, X_cont], axis=1)
    return X


def train_model(X_train, y_train):
    """
    Trains a Random Forest model and return it
    """
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(
        model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    logging.info(f'Accuracy: {np.mean(scores)}({np.std(scores)} std)')
    return model


def compute_metrics(y_true, y_pred):
    """
    Compute precision, recall and F1
    """
    fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Use model to infer
    """
    y_pred = model.predict(X)
    return y_pred
