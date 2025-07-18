from azure.ai.ml import MLClient, load_job
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential = DefaultAzureCredential(),
    subscription_id = "2c63df35-0b4a-4a70-836d-cf53462ec189",
    resource_group_name="manoj_rg",
    workspace_name="Customer-Churn-Classification"
)

job = load_job(source="Training/jobs/train-job.yml")
sumbitted_job = ml_client.jobs.create_or_update(job)
print(f"Submitted Job: {sumbitted_job.name}")
