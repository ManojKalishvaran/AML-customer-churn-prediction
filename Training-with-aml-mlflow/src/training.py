import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from preprocessing import Preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_file(file_path: str) -> pd.DataFrame:
    # Adjust sep if needed; using CSV default here
    return pd.read_csv(file_path, sep=",")

def split_data(data):
    X = data.drop(columns=['Churn'], axis=1)
    y = data['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------------------------------------------
# Train / evaluate
# -------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    params = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}
    pipe = Pipeline(
        [
            ("preprocess", Preprocess()),
            ("model", RandomForestClassifier(**params)),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
    }
    return pipe, params, scores


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main(args):
    df = load_file(args.train_file)
    X_train, X_test, y_train, y_test = split_data(df)

    # Single MLflow run; Azure ML auto-configures tracking when run as a job.
    with mlflow.start_run() as run:
        # Lineage tags from job submission (optional but recommended)
        if args.data_name:
            mlflow.set_tag("data_asset_name", args.data_name)
        if args.data_version:
            mlflow.set_tag("data_asset_version", args.data_version)

        model, params, scores = train_and_evaluate(
            X_train, X_test, y_train, y_test, args.n_estimators
        )

        # Log
        mlflow.log_params(params)
        mlflow.log_metrics(scores)
        mlflow.log_param("label_col", args.label_col)

        # Persist metrics JSON
        Path("artifacts").mkdir(exist_ok=True)
        with open("artifacts/metrics.json", "w") as f:
            json.dump(scores, f)
        mlflow.log_artifact("artifacts/metrics.json", artifact_path="eval")

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("Run complete. id=%s", run.info.run_id)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True, help="Path to mounted training file.")
    p.add_argument("--data-name", default=None, help="Data Asset name (for lineage tag).")
    p.add_argument("--data-version", default=None, help="Data Asset version.")
    p.add_argument("--label-col", default="Churn", help="Name of the target column.")
    p.add_argument("--n-estimators", type=int, default=100)
    args = p.parse_args()
    main(args)
