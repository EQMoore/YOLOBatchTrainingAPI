# YOLO Batch Training API

A FastAPI service that abstracts training a YOLOv8 model on Google Cloud. A
client uploads a dataset ZIP and a model name; the API stores the ZIP in Cloud
Storage and submits a Vertex AI custom training job that runs the trainer
container. The trainer downloads the dataset, trains with Ultralytics YOLOv8,
exports to ONNX, attempts dynamic quantization, and uploads the artifacts back
to Cloud Storage.

## Architecture

```
client ──POST /train_yolo──▶ FastAPI (main.py)
                                │  upload ZIP → gs://$BUCKET_NAME/{user_id}/{model}.zip
                                │  submit Vertex AI CustomContainerTrainingJob
                                ▼
                        Vertex AI runs container_image/  (trainer.DOCKERFILE)
                                │  download ZIP, extract, YOLO train, export ONNX, quantize
                                ▼
                        gs://$BUCKET_NAME/{user_id}/{model}/
                            ├── final_model.pt
                            ├── final_model.onnx
                            └── final_model.quant.onnx
```

`main.py` / `gcs_util.py` — the API service.
`container_image/` — the trainer, built as its own image and run by Vertex AI.

## Environment variables

Required by the API service:

| Variable | Description |
| --- | --- |
| `BUCKET_NAME` | GCS bucket for datasets and artifacts (bucket name only, no `gs://`) |
| `PROJECT_ID` | GCP project id |
| `REGION` | Region for Vertex AI (e.g. `us-central1`) |
| `VERTEX_CONTAINER_URI` | Image URI of the trainer container (e.g. `REGION-docker.pkg.dev/PROJECT/REPO/yolo-trainer:TAG`) |

The trainer container needs `BUCKET_NAME` (and `PROJECT_ID` for Vertex).

## API

### `POST /train_yolo` → `202 Accepted`

The dataset ZIP is sent as a `multipart/form-data` body; every other parameter
is a **query-string** parameter.

Multipart body:

| Field | Type | Notes |
| --- | --- | --- |
| `dataset` | file | ZIP containing an Ultralytics-format dataset with a `data.yaml` |

Query parameters:

| Param | Type | Default | Notes |
| --- | --- | --- | --- |
| `model` | string | — | Name for this model; used in the output path |
| `arch` | string | `yolov8n` | Base checkpoint to fine-tune (`yolov8n`, `yolov8s`, …) |
| `epochs` | int | `10` | |
| `batch` | int | `16` | |
| `user_id` | string | `default` | Owner id; used as the output path prefix |

Example:

```bash
curl -X POST "http://localhost:8000/train_yolo?model=car-detector&epochs=10&user_id=alice" \
  -F "dataset=@data.zip"
```

The ZIP is uploaded to `gs://$BUCKET_NAME/{user_id}/{model}.zip`. If
`{user_id}/{model}` already exists the request returns `409`.

Response:

```json
{ "job_id": "yolo-train-<uuid>", "user_id": "...", "model": "...", "status": "submitted" }
```

The job runs asynchronously on Vertex AI; the response returns as soon as the
job is created.

### `GET /get_models?user_id=<id>`

Returns the list of GCS object names under the `user_id` prefix.

### `GET /download_model?user_id=<id>&model_name=<name>`

Returns the list of GCS object names under `{user_id}/{model_name}`.

> Note: `download_model` / `download_model_file` currently list objects rather
> than streaming a file. Streaming the artifact is not finished yet.

## Trainer container

Built from `container_image/trainer.DOCKERFILE` with `container_image/` as the
build context:

```
docker build container_image --file container_image/trainer.DOCKERFILE --tag yolo-trainer:local
```

CLI (`container_image/main.py`), invoked by Vertex AI:

```
python main.py \
  --dataset_zip=gs://BUCKET/USER/MODEL.zip \
  --user_id=USER \
  --model=MODEL \
  --arch=yolov8n \
  --epochs=10 \
  --batch=16
```

It uploads `final_model.pt`, `final_model.onnx` and `final_model.quant.onnx`
(whichever were produced) to `gs://$BUCKET_NAME/{user_id}/{model}/`. It exits
non-zero if the ZIP is invalid or training produces no artifacts.

## Local development

```
pip install -r requirements.txt
uvicorn main:app --reload
```
