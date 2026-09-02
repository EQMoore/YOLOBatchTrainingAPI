# YOLO Batch Training API

A FastAPI service that abstracts fine-tuning a YOLOv8 model on Google Cloud. A
client uploads a dataset ZIP and a model name; the API stores the ZIP in Cloud
Storage and submits a Vertex AI custom training job that runs the trainer
container. The trainer downloads the dataset, trains with Ultralytics YOLOv8,
exports to ONNX, attempts dynamic quantization, and uploads the artifacts back
to Cloud Storage. Training runs asynchronously — `POST /train_yolo` returns as
soon as the job is created.

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

## Repository layout

| Path | What it is |
| --- | --- |
| `main.py` | FastAPI app — all HTTP endpoints |
| `auth.py` | Bearer-token auth; resolves a token to a user id |
| `gcs_util.py` | Cloud Storage helpers and Vertex AI job submission |
| `api.DOCKERFILE` | Image for the API service |
| `container_image/` | The trainer — its own image, run by Vertex AI (`trainer.DOCKERFILE`) |
| `infra/` | One-time GCP provisioning ([`infra/README.md`](infra/README.md)) |
| `tests/` | Unit tests for the API service |

## Environment variables

Required by the API service:

| Variable | Description |
| --- | --- |
| `API_TOKENS` | `token:user` pairs, comma-separated (e.g. `tok_abc:alice,tok_def:bob`) |
| `BUCKET_NAME` | GCS bucket for datasets and artifacts (bucket name only, no `gs://`) |
| `PROJECT_ID` | GCP project id |
| `REGION` | Region for Vertex AI (e.g. `us-central1`) |
| `VERTEX_CONTAINER_URI` | Image URI of the trainer container (e.g. `REGION-docker.pkg.dev/PROJECT/REPO/yolo-trainer:TAG`) |
| `PORT` | Port the API listens on (default `8080`; injected by Cloud Run) |

The trainer container needs `BUCKET_NAME` (and runs with a Vertex service
identity that can read/write the bucket).

## Authentication

Every endpoint requires a bearer token:

```
Authorization: Bearer <token>
```

The token is matched (constant-time) against `API_TOKENS` and resolved to a user
id. That user id is the owner prefix for every storage path — it is **never**
taken from the request. Lookups are scoped to the whole `{user_id}/` path
segment, so user `alice` cannot see objects belonging to `alice-corp`. A missing
or unknown token gets `401`.

## API

### `POST /train_yolo` → `202 Accepted`

`multipart/form-data` body:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `dataset` | file | — | ZIP containing an Ultralytics-format dataset with a `data.yaml` |
| `model` | string | — | Name for this model; used in the output path (`[A-Za-z0-9._-]`, ≤ 64 chars) |
| `arch` | string | `yolov8n` | Base checkpoint to fine-tune (`yolov8n`, `yolov8s`, …) |
| `epochs` | int | `10` | |
| `batch` | int | `16` | |

```bash
curl -X POST "http://localhost:8000/train_yolo" \
  -H "Authorization: Bearer tok_abc" \
  -F "dataset=@data.zip" \
  -F "model=car-detector" \
  -F "epochs=10"
```

The ZIP is uploaded to `gs://$BUCKET_NAME/{user_id}/{model}.zip`, then a Vertex
AI job is submitted.

Response:

```json
{ "job_id": "yolo-train-<uuid>", "user_id": "alice", "model": "car-detector", "status": "submitted" }
```

Errors: `400` (no dataset, or `model` / `arch` fails validation), `401`
(bad token), `409` (`{user_id}/{model}` already has a dataset or artifacts).

### `GET /get_models`

Returns the list of GCS object names under `{user_id}/` — every dataset and
artifact the caller owns.

### `GET /download_model?model_name=<name>`

Returns the list of GCS object names under `{user_id}/{model_name}/`. This
**lists** a model's files; it does not return their contents — use
`/download_model_file` for that.

### `GET /download_model_file?model=<name>&artifact=<name>`

Streams one artifact as a file download. `artifact` must be one of
`final_model.pt`, `final_model.onnx`, `final_model.quant.onnx` (default:
`final_model.quant.onnx`). `404` if that artifact does not exist.

## Running the API

### Locally

```bash
pip install -r requirements.txt
export API_TOKENS="tok_dev:me"
uvicorn main:app --reload
```

### As a container

Built from `api.DOCKERFILE` with the repo root as the build context:

```bash
docker build . --file api.DOCKERFILE --tag yolo-api:local
docker run --rm -p 8000:8080 \
  -e API_TOKENS="tok_dev:me" \
  -e BUCKET_NAME=... -e PROJECT_ID=... -e REGION=us-central1 \
  -e VERTEX_CONTAINER_URI=... \
  yolo-api:local
```

The container listens on `$PORT` (default `8080`) as a non-root user. GCP
credentials come from the environment — Workload Identity on Cloud Run / GKE, or
a mounted key via `GOOGLE_APPLICATION_CREDENTIALS`.

## Trainer container

Built from `container_image/trainer.DOCKERFILE` with `container_image/` as the
build context:

```bash
docker build container_image --file container_image/trainer.DOCKERFILE --tag yolo-trainer:local
```

CLI (`container_image/main.py`), invoked by Vertex AI:

```
python main.py \
  --dataset_zip=gs://BUCKET/USER/MODEL.zip \
  --user_id=USER --model=MODEL \
  --arch=yolov8n --epochs=10 --batch=16
```

It uploads whichever of `final_model.pt`, `final_model.onnx`,
`final_model.quant.onnx` were produced to `gs://$BUCKET_NAME/{user_id}/{model}/`,
and exits non-zero if the ZIP is invalid or training produces no artifacts.

## GCP setup

One-time provisioning — Artifact Registry, the bucket, service accounts, and
Workload Identity Federation for CI — is scripted in
[`infra/gcp_setup.sh`](infra/README.md). It is idempotent.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Unit tests cover the API service (`main.py`, `auth.py`, `gcs_util.py`) with no
network or GCP access — the Cloud Storage and Vertex AI clients are stubbed.
`pytest` is configured (in `pyproject.toml`) to fail under **100 % line + branch
coverage**.

## CI/CD

| Workflow | Trigger | Does |
| --- | --- | --- |
| **Tests** (`.github/workflows/tests.yml`) | every push, PRs to `main` | runs `pytest` with the 100 % coverage gate |
| **Docker Image CI** (`.github/workflows/docker-image.yml`) | every push, PRs to `main` | builds both `api.DOCKERFILE` and `container_image/trainer.DOCKERFILE` |

`main` is a protected branch: the **Tests** check must pass before a pull
request can be merged.

## Known limitations

- No deploy step — CI builds the images but does not push or deploy them.
- No job-status endpoint; once submitted, training progress is only visible in
  the Vertex AI console.
- `download_model_file` buffers the artifact in memory before sending it.
- No request-size limit on the dataset upload.
