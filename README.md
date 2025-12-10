Project: YOLO Training via API that asigns a group of google cloud Container's to train models for you

A FastAPI app (`main.py`) accepts a dataset ZIP upload and a model name. It uploads the ZIP to GCS and submits a managed training job (Vertex AI CustomContainerTrainingJob) which runs the trainer inside a container.

The trainer code is in `vm_image/` and is designed to run inside the training container. It downloads the dataset from GCS, extracts it, runs training using Ultralytics YOLOv8, exports to ONNX and attempts ONNX quantization, then uploads artifacts back to GCS.

Environment variables
`BUCKET_NAME` - GCS bucket used to store datasets and artifacts
`PROJECT_ID` - GCP project id
`REGION` - GCP region for Vertex AI
`VERTEXT_CONTAINER_URI` - Container image URI to use for Vertex AI training job

How to train a model with a post request

POST `/train_yolo` with form-data: `dataset` (file), `model` (string), optional `epochs` and `batch`, `user_id`.

The API uploads the ZIP to `gs://{BUCKET_NAME}/{user_id}/{model}.zip` and calls `gcs_util.submit_training_job(...)`

Vertex AI launches the container. The container runs the trainer script. Then the trainer downloads the ZIP, extracts, trains, exports ONNX, quantizes, and uploads artifacts back to `gs://{BUCKET_NAME}/{user_id}/{model}/`

TODO: Make the API listen for an gs upload and send a progress update to the user
