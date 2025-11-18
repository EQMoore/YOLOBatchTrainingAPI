import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import google_util

app = FastAPI()

@app.post("/train_yolo")
def train_yolo(dataset: UploadFile, model: str, epochs: int, batch: int, user_id:str):
    temp_name = f"/tmp/{uuid.uuid4()}.zip"
    blob_path = user_id+"/"+model
    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(dataset.file, buffer)
        #TODO: add .env file
    if(google_util.check_gcs_unique_name(blob_path)):
        return "Model name already in use"
    gcs_path = google_util.upload_to_gcs(temp_name, blob_path)
    google_util.submit_vertex_training_job(gcs_path,blob_path,epochs,batch)

@app.get("/get_models")
def get_models(user_id):
    return google_util.get_user_models(user_id)
    
    
