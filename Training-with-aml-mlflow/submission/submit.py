from azure.ai.ml import command, Input, MLClient
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential
import logging
logging.basicConfig(level=logging.INFO)

# ml_client = MLClient(
# 	DefaultAzureCredential(),

# )

sp_client = ""
sp_client_sceret = ""
sp_tenent = "add sp tenant"
from azure.identity import ClientSecretCredential

credential = ClientSecretCredential(
	sp_tenent, sp_client, sp_client_sceret
)

ml_client = MLClient.from_config(credential)

data_version = "1"

ds = ml_client.data.get(name="training-data", version=data_version)

job = command(
	code="./Training-with-aml-mlflow/src",
	command=[""+i for i in (
		"python training2.py ",
		"--train-file ${{inputs.train_file}} ",
		f"--data-name {ds.name}",
		f"--data-version {ds.version} "
	)][0],
	inputs={
		"train_file":Input(
			type=AssetTypes.URI_FILE,
			path = ds.id,
			mode = InputOutputModes.RO_MOUNT
		)
	}, 
	# environment="azureml:sklearn1.5@latest",
	environment="azureml://registries/azureml/environments/sklearn-1.5/labels/latest",
	compute="cpu-cluster",
	experiment_name="Churn-pred-from-thusiba",
	display_name=f"Churn-train-data-v{ds.version}"
)

# inputs={
# "train_file":Input(
# 	type=AssetTypes.URI_FILE,
# 	path = ds.id,
# 	mode = InputOutputModes.RO_MOUNT
# )}

# print(f"-------------------------input path:{inputs["train_file"]["path"]}")

returned_job = ml_client.jobs.create_or_update(job)