import os
from google.cloud import storage, aiplatform
import uuid
import os

BUCKET_NAME = os.getenv("BUCKET_NAME")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
VERTEXT_CONTAINER_URI = os.getenv("VERTEXT_CONTAINER_URI")

def get_user_models(user_id:str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=user_id))
    return [b.name for b in blobs]

def check_gcs_unique_name(name: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=name))
    return len(blobs) > 0

def upload_to_gcs(local_path: str, blob_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/{blob_path}"


def submit_training_job(dataset_gcs_path: str, model_name: str, epochs: int, batch: int):
    aiplatform.init(project=PROJECT_ID, location=REGION)

    job_id = f"yolo-train-{uuid.uuid4()}"

    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_id,
        container_uri=os.getenv("VERTEXT_CONTAINER_URI")
    )
    args = [
        f"--dataset_zip={dataset_gcs_path}",
        f"--model={model_name}",
        f"--epochs={epochs}",
        f"--batch={batch}",
    ]

    job.run(
        args=args,
        replica_count=1,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        base_output_dir=f"gs://{BUCKET_NAME}/training_outputs/{job_id}",
    )

    return job_id
