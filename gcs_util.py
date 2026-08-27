import os
from google.cloud import storage, aiplatform
import uuid
import os

BUCKET_NAME = os.getenv("BUCKET_NAME")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
VERTEX_CONTAINER_URI = os.getenv("VERTEX_CONTAINER_URI", "default")

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

def submit_training_job(dataset_gcs_uri: str, user_id: str, model_name: str,
                        epochs: int, batch: int, arch: str = "yolov8n") -> str:
    if not all([BUCKET_NAME, PROJECT_ID, REGION]) or VERTEX_CONTAINER_URI == "default":
        raise RuntimeError(
            "Missing required environment: BUCKET_NAME, PROJECT_ID, REGION, "
            "VERTEX_CONTAINER_URI"
        )

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}",
    )
    job_id = f"yolo-train-{uuid.uuid4()}"

    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_id,
        container_uri=VERTEX_CONTAINER_URI,
    )
    #trainer CLI: container_image/main.py -- all args are --flag=value strings
    args = [
        f"--dataset_zip={dataset_gcs_uri}",
        f"--user_id={user_id}",
        f"--model={model_name}",
        f"--arch={arch}",
        f"--epochs={epochs}",
        f"--batch={batch}",
    ]

    #submit() returns once the job is created; run() would block until training ends
    job.submit(
        args=args,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        base_output_dir=f"gs://{BUCKET_NAME}/training_outputs/{job_id}",
    )

    return job_id
