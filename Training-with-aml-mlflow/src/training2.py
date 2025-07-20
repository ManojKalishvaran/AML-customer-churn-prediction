import argparse, os, json
import mlflow
import mlflow.sklearn
import pandas as pd
import pathlib as path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import Preprocess
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import logging
logging.basicConfig(level=logging.INFO)

def load_file(file_path: str) -> pd.DataFrame:
	logging.info(f"File path: {file_path}")
	return pd.read_csv(file_path, sep=",")

def split_data(data):
    X = data.drop(columns=['Churn'], axis=1)
    y = data['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_and_evaluate(X_train, X_test, y_train, y_test):
	
	with mlflow.start_run() as run:
		parameters = {"n_estimators":100, "random_state":42}
		pipe = Pipeline([
			("preprocess", Preprocess()),
			("model", RandomForestClassifier(**parameters))
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
		
		mlflow.log_params(parameters)
		mlflow.log_metrics(scores)
		mlflow.sklearn.log_model(pipe, artifact_path="model")

def main(args):

    data = load_file(args.train_file)
    X_train, X_test, y_train, y_test = split_data(data)
    model, scores = train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__=="__main__":
	p = argparse.ArgumentParser()
	p.add_argument("--train-file", required=True)
	p.add_argument("--data-name", default=None)
	p.add_argument("--data-version", default=None)
	args = p.parse_args()
	main(args)