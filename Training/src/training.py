import pandas as pd
import argparse
import os
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from preprocessing import Preprocess

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to input dataset")
    parser.add_argument("--output_model", type=str, help="Path to save the model")
    parser.add_argument("--base_line", type=int, default=70, help="Baseline accuracy for storing model")
    args = parser.parse_args()
    return args

def load_data(input_path):
    df = pd.read_csv(input_path)
    return df

def split_data(data):
    X = data.drop(columns=['Churn'], axis=1)
    y = data['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    pipe = Pipeline([
        ("preprocess", Preprocess()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }
    return pipe, scores

def main():
    args = parse_args()

    data = load_data(args.input_data)
    X_train, X_test, y_train, y_test = split_data(data)
    model, scores = train_and_evaluate(X_train, X_test, y_train, y_test)

    logging.info(f"Evaluation Metrics: {scores}")

    if scores['accuracy'] >= args.base_line / 100:
        os.makedirs(args.output_model, exist_ok=True)
        model_path = os.path.join(args.output_model, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"✅ Model saved to {model_path}")
    else:
        logging.info("❌ Model did not meet baseline accuracy. Not saved.")

if __name__ == "__main__":
    main()
