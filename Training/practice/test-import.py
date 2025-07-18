import cloudpickle
import sys

# Step 1: Add the correct path
sys.path.append(r"D:\Azure MLOps learning\Custom project\customer churn prediction\AML-customer-churn\AML-customer-churn-prediction\Training")

# Step 2: Import your transformer
from preprocessing import Preprocess

# Step 3: Load using cloudpickle
with open(r"D:\Azure MLOps learning\Custom project\customer churn prediction\AML-customer-churn\AML-customer-churn-prediction\Training\practice\downloaded_model\named-outputs\trained_model\model.pkl", "rb") as f:
    model = cloudpickle.load(f)

# Step 4: Don't print `model` directly!
print("✅ Model loaded:", type(model))
print(model.predict())

# Optional: make a prediction (if you have test data)
# import pandas as pd
# X_sample = pd.read_csv("sample.csv")  # should match training schema
# print(model.predict(X_sample))
