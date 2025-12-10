import os
from google.cloud import storage
from google.cloud import bigquery

project_number = os.environ["CLOUD_ML_PROJECT_ID"]

client = bigquery.Client(project=project_number)

def _parse_gs_uri(gs_uri: str):
    if not gs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    parts = gs_uri[5:].split('/', 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ''
    return bucket, blob

def download_from_gcs(local_path: str, gs_uri: str) -> str:
    client = storage.Client()
    bucket_name, blob_path = _parse_gs_uri(gs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    return local_path

def upload_model(local_path: str) -> str:
    client = storage.Client()
    bucket_name, blob_path = _parse_gs_uri(os.getenv('AIP_MODEL_DIR', 'error'))
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_path}"