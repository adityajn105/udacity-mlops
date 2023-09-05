"""
Evaluation for slices
"""

import pandas as pd
import src.utils as utils
from sklearn.model_selection import train_test_split
import logging
from joblib import load


def evaluate_slices(path="data/cleaned_census.csv"):
    """
    Evaluate model
    """
    df = pd.read_csv(path)
    _, test = train_test_split(df, test_size=0.2)

    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/label_binarizer.joblib")

    slice_eval_logs = []
    for cat_var in utils.categorical_features:
        for category in test[cat_var].unique():
            df_sliced = test[test[cat_var] == category]

            X_test, y_test = utils.process_test_data(
                df_sliced, "salary", encoder, lb)
            y_pred = model.predict(X_test)
            precision, recall, f1 = utils.compute_metrics(y_test, y_pred)
            log_line = f"[{cat_var}-> {category}] \
                | Precision: {precision}, Recall: {recall}, F1: {f1}"
            logging.info(log_line)
            slice_eval_logs.append(log_line)

    with open("data/sliced_evaluation.txt", "w") as fp:
        fp.write("\n".join(slice_eval_logs))
