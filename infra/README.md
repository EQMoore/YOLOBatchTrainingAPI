# GCP setup

`gcp_setup.sh` provisions everything the API and its CI need. Run it once, from
a machine authenticated with an account that has Owner (or an equivalent set of
admin roles) on the project.

```bash
PROJECT_ID=my-project \
BUCKET_NAME=my-yolo-bucket \
GITHUB_REPO=my-org/YOLOBatchTrainingAPI \
REGION=us-central1 \
  ./infra/gcp_setup.sh
```

It is idempotent — re-running it is safe.

## What it creates

| Resource | Purpose |
| --- | --- |
| Enabled APIs | Artifact Registry, Vertex AI, Cloud Storage, Cloud Run, IAM Credentials, STS, Secret Manager |
| Artifact Registry repo `yolo` | Holds the trainer and API images |
| GCS bucket | Datasets, training artifacts, Vertex staging |
| Service account `yolo-api` | Cloud Run runtime identity — `aiplatform.user` + `storage.objectAdmin` on the bucket |
| Vertex custom-code service agent | Granted `storage.objectAdmin` on the bucket (it runs the training container) |
| Service account `yolo-ci` | GitHub Actions identity — `artifactregistry.writer`, `run.admin`, and `serviceAccountUser` on `yolo-api` |
| Workload Identity pool/provider `github` | Lets GitHub Actions impersonate `yolo-ci` with no key file; locked to `GITHUB_REPO` |

## After running it

1. The script prints a block of values. Add them as **GitHub Actions repository
   variables** (`Settings → Secrets and variables → Actions → Variables`).
2. Create the API token secret:
   ```bash
   printf 'tok_xxx:alice' | gcloud secrets create api-tokens --data-file=-
   ```
3. The trainer image URI (used as `VERTEX_CONTAINER_URI`) is:
   ```
   REGION-docker.pkg.dev/PROJECT_ID/yolo/yolo-trainer:latest
   ```

## Teardown

```bash
gcloud artifacts repositories delete yolo --location=REGION
gcloud storage rm -r gs://BUCKET_NAME
gcloud iam service-accounts delete yolo-api@PROJECT_ID.iam.gserviceaccount.com
gcloud iam service-accounts delete yolo-ci@PROJECT_ID.iam.gserviceaccount.com
gcloud iam workload-identity-pools delete github --location=global
```
