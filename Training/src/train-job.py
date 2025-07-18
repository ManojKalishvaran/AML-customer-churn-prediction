from azure.ai.ml import MLClient, load_component, Input
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

# Connect to your workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id = "2c63df35-0b4a-4a70-836d-cf53462ec189",
    resource_group_name="manoj_rg",
    workspace_name="Customer-Churn-Classification"
)

# Load the command component from YAML
train_component = load_component(source="./Training/components/train.yml")
print("After training component")

# Define the pipeline using DSL
@pipeline(
    compute="cpu-cluster",  # Use your AML cluster name
    experiment_name="churn-pipeline"
)
def churn_training_pipeline(input_data, base_line):
    train_step = train_component(
        input_data=input_data,
        base_line=base_line
    )
    return {
        "trained_model": train_step.outputs.output_model
    }

# Instantiate the pipeline job
pipeline_job = churn_training_pipeline(
    input_data=Input(type="uri_file", path="azureml:training-data:1"),  # Registered dataset
    base_line=70
)

# Submit the job
submitted_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"✅ Pipeline submitted. Name: {submitted_job.name}")
