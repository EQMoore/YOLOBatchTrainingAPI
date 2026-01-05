import os
from google.cloud import storage
from google.cloud import bigquery

project_number = os.environ["PROJECT_ID"]

client = bigquery.Client(project=project_number)

def _parse_gs_uri(gs_uri: str):
    #check if it starts with gs://
    if not gs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    #read from the 5th ind. onwards (everything after gs://), and then split by the first /
    parts = gs_uri[5:].split('/', 1)
    #the bucket is the first part of the gcs_uri, bucket/blob, parts = ["bucket", "blob"]
    bucket = parts[0]
    #put the blob as the 1st part, if there isn't a blob part to the url return and empty string
    if (len(parts) > 1):
        blob = parts[1]
        return bucket, blob
    else:
        raise ValueError("GCS URI must contain a blob path")

def download_from_gcs(local_path: str, gs_uri: str) -> str:
    client = storage.Client()
    bucket_name, blob_path = _parse_gs_uri(gs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    return local_path

def upload_model(local_path: str) -> str:
    client = storage.Client()
    bucket_name, blob_path = _parse_gs_uri(os.getenv('BUCKET_NAME', 'error'))
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_path}"