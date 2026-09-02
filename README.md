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
| `API_TOKENS` | `token:user` pairs, comma-separated (e.g. `tok_abc:alice,tok_def:bob`) |

The trainer container needs `BUCKET_NAME` (and `PROJECT_ID` for Vertex).

## GCP setup

One-time provisioning (Artifact Registry, the bucket, service accounts, and
Workload Identity Federation for CI) is scripted in
[`infra/gcp_setup.sh`](infra/README.md).

## Authentication

Every endpoint requires a bearer token:

```
Authorization: Bearer <token>
```

The token is matched against `API_TOKENS` and resolved to a user id. That user
id is the owner prefix for all storage paths — it is **never** taken from the
request, so a caller cannot act on another user's models. Requests with a
missing or unknown token get `401`.

## API

### `POST /train_yolo` → `202 Accepted`

`multipart/form-data` body:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `dataset` | file | — | ZIP containing an Ultralytics-format dataset with a `data.yaml` |
| `model` | string | — | Name for this model; used in the output path (`[A-Za-z0-9._-]`, ≤64) |
| `arch` | string | `yolov8n` | Base checkpoint to fine-tune (`yolov8n`, `yolov8s`, …) |
| `epochs` | int | `10` | |
| `batch` | int | `16` | |

Example:

```bash
curl -X POST "http://localhost:8000/train_yolo" \
  -H "Authorization: Bearer tok_abc" \
  -F "dataset=@data.zip" \
  -F "model=car-detector" \
  -F "epochs=10"
```

The ZIP is uploaded to `gs://$BUCKET_NAME/{user_id}/{model}.zip`. If
`{user_id}/{model}` already exists the request returns `409`.

Response:

```json
{ "job_id": "yolo-train-<uuid>", "user_id": "...", "model": "...", "status": "submitted" }
```

The job runs asynchronously on Vertex AI; the response returns as soon as the
job is created.

### `GET /get_models`

Returns the list of GCS object names owned by the caller.

### `GET /download_model?model_name=<name>`

Returns the list of GCS object names under `{user_id}/{model_name}`.

### `GET /download_model_file?model=<name>&artifact=<name>`

Streams one artifact (`final_model.pt`, `final_model.onnx`, or
`final_model.quant.onnx`; defaults to the quantized ONNX) as a file download.

## API container

Built from `api.DOCKERFILE` with the repo root as the build context:

```
docker build . --file api.DOCKERFILE --tag yolo-api:local
docker run --rm -p 8000:8080 \
  -e API_TOKENS="tok_dev:me" \
  -e BUCKET_NAME=... -e PROJECT_ID=... -e REGION=us-central1 \
  -e VERTEX_CONTAINER_URI=... \
  yolo-api:local
```

The container listens on `$PORT` (default `8080`). GCP credentials are picked up
from the environment (Workload Identity on Cloud Run / GKE, or a mounted service
account key via `GOOGLE_APPLICATION_CREDENTIALS`).

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
export API_TOKENS="tok_dev:me"
uvicorn main:app --reload
```
