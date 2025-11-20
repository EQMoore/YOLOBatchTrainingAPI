import tempfile
import shutil
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import os
import io
from google.cloud import storage
import gcs_util
import compute_util

app = FastAPI()


@app.post("/train_yolo")
def train_yolo(dataset: UploadFile, model: str, epochs: int = 10, batch: int = 16, user_id: str = "default"):
    if not dataset.filename:
        raise HTTPException(status_code=400, detail="No dataset uploaded")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = temp_file.name
        dataset.file.seek(0)
        shutil.copyfileobj(dataset.file, temp_file)

    try:
        blob_path = f"{user_id}/{model}.zip"
        if gcs_util.check_gcs_unique_name(f"{user_id}/{model}"):
            os.remove(temp_path)
            raise HTTPException(status_code=409, detail="Model name already in use")

        gcs_path = gcs_util.upload_to_gcs(temp_path, blob_path)

        from compute_util import create_training_vm

        job_id = create_training_vm(gcs_path, f"{user_id}/{model}", epochs, batch)
        return {"job_id": job_id}
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


@app.get("/get_models")
def get_models(user_id: str):
    return gcs_util.get_user_models(user_id)


@app.get("/download_model")
def download_model(user_id: str, model_name: str):
    return gcs_util.get_user_models(f"{user_id}/{model_name}")


def stream_blob(bucket_name: str, blob_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    stream = io.BytesIO()
    blob.download_to_file(stream)
    stream.seek(0)
    return StreamingResponse(stream, media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename={os.path.basename(blob_name)}"})


@app.get("/job_status")
def job_status(operation: str = None, instance: str = None):
    if operation:
        return compute_util.get_zone_operation_status(operation)
    if instance:
        return compute_util.get_instance_status(instance)
    raise HTTPException(status_code=400, detail="Provide operation or instance")


@app.get("/download_model_file")
def download_model_file(user_id: str, model: str, artifact: str = "final_model.quant.onnx"):
    blob_name = f"{user_id}/{model}/{artifact}"
    return stream_blob(os.getenv("BUCKET_NAME"), blob_name)