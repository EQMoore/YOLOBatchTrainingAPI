import tempfile
import shutil
from fastapi import FastAPI, UploadFile, HTTPException
import gcs_util
import os

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