import os
from google.cloud import storage, aiplatform
import uuid

PROJECT_ID = "PROJECT_ID"
BUCKET_NAME = "BUCKET_NAME"
REGION = "us-central1"
VERTEX_CONTAINER_URI = "us-docker.pkg.dev/YOUR_PROJECT_ID/yolo-trainer/yolo:latest"

def get_user_models(user_id:str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=user_id, versions=True)
    return(blobs!=None)

def check_gcs_unique_name(name:str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=name, versions=True)
    return(blobs!=None)

def upload_to_gcs(local_path: str, blob_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/{blob_path}"


def submit_vertex_training_job(dataset_gcs_path: str, model_name: str, epochs: int, batch: int):
    aiplatform.init(project=PROJECT_ID, location=REGION)

    job_id = f"yolo-train-{uuid.uuid4()}"

    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_id,
        container_uri=VERTEX_CONTAINER_URI,
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
