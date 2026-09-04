import os
from google.cloud import storage, aiplatform
import uuid
import os

BUCKET_NAME = os.getenv("BUCKET_NAME")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
VERTEX_CONTAINER_URI = os.getenv("VERTEX_CONTAINER_URI", "default")

def get_user_models(prefix: str):
    #match on a whole path segment: get_user_models("alice") must not leak
    #"alice-corp/..." objects. Callers pass a user id or "{user_id}/{model}".
    prefix = prefix.rstrip("/") + "/"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return [b.name for b in blobs]

def check_gcs_unique_name(name: str):
    #True if the model name is taken: its dataset zip or any artifact exists.
    #Segment-scoped, so "car" does not collide with "car-detector".
    zip_name = f"{name}.zip"
    artifact_prefix = f"{name}/"
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    for blob in bucket.list_blobs(prefix=name):
        if blob.name == zip_name or blob.name.startswith(artifact_prefix):
            return True
    return False

def upload_to_gcs(local_path: str, blob_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/{blob_path}"

def delete_from_gcs(blob_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    bucket.blob(blob_path).delete()

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

    submit_kwargs = dict(
        args=args,
        machine_type=os.getenv("TRAIN_MACHINE_TYPE", "n1-standard-8"),
        base_output_dir=f"gs://{BUCKET_NAME}/training_outputs/{job_id}",
    )
    #attach a GPU only when TRAIN_ACCELERATOR_TYPE is set (e.g. NVIDIA_TESLA_T4);
    #unset -> CPU-only, which needs no GPU quota
    accelerator = os.getenv("TRAIN_ACCELERATOR_TYPE")
    if accelerator:
        submit_kwargs["accelerator_type"] = accelerator
        submit_kwargs["accelerator_count"] = int(os.getenv("TRAIN_ACCELERATOR_COUNT", "1"))

    #submit() returns once the job is created; run() would block until training ends
    job.submit(**submit_kwargs)

    return job_id
