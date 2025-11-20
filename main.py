import tempfile
import shutil
from fastapi import FastAPI, UploadFile
import gcs_util

app = FastAPI()

@app.post("/train_yolo")
def train_yolo(dataset: UploadFile, model: str, epochs: int, batch: int, user_id:str):
    with tempfile.NamedTemporaryFile(suffix=".zip") as temp_file:
        temp_name = temp_file.name
        shutil.copyfileobj(dataset.file, temp_name)
        blob_path = f"{user_id}/{model}"
        if(gcs_util.check_gcs_unique_name(blob_path)):
            return "Model name already in use"
        gcs_path = gcs_util.upload_to_gcs(temp_name, blob_path)
        gcs_util.submit_training_job(gcs_path,blob_path,epochs,batch)
    
@app.get("/get_models")
def get_models(user_id:str):
    return gcs_util.get_user_models(user_id)

#TODO
@app.get("download_model")
def download_model(user_id:str, model_name:str):
    return gcs_util.get_user_models(user_id)