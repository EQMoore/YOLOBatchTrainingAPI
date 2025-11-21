import io
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import types

sys.modules.setdefault('google', types.ModuleType('google'))
sys.modules.setdefault('google.cloud', types.ModuleType('google.cloud'))
sys.modules.setdefault('google.cloud.storage', types.ModuleType('google.cloud.storage'))
sys.modules.setdefault('google.cloud.aiplatform', types.ModuleType('google.cloud.aiplatform'))
sys.modules.setdefault('googleapiclient', types.ModuleType('googleapiclient'))
sys.modules.setdefault('googleapiclient.discovery', types.ModuleType('googleapiclient.discovery'))

import gcs_util
import compute_util
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_job_status_bad_request():
    r = client.get("/job_status")
    assert r.status_code == 400


def test_get_models(monkeypatch):
    def fake_get_user_models(user_id):
        return [f"{user_id}/m1"]

    monkeypatch.setattr(gcs_util, 'get_user_models', fake_get_user_models)
    r = client.get("/get_models", params={"user_id": "u1"})
    assert r.status_code == 200
    assert r.json() == ["u1/m1"]


def test_download_model_file(monkeypatch):
    def fake_stream_blob(bucket, blob):
        from fastapi.responses import StreamingResponse
        return StreamingResponse(io.BytesIO(b"data"), media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename=test.bin"})

    monkeypatch.setattr(main, 'stream_blob', fake_stream_blob)
    r = client.get("/download_model_file", params={"user_id": "u1", "model": "m1", "artifact": "a.onnx"})
    assert r.status_code == 200
    assert r.content == b"data"
    assert r.headers.get("content-disposition") is not None


def test_train_yolo_success(monkeypatch):
    monkeypatch.setattr(gcs_util, 'check_gcs_unique_name', lambda name: False)
    monkeypatch.setattr(gcs_util, 'upload_to_gcs', lambda local_path, blob_path: f"gs://fake-bucket/{blob_path}")
    monkeypatch.setattr(compute_util, 'create_training_vm', lambda gcs_path, model_path, epochs, batch: {"operation": "op1", "instance": "inst1"})

    data = {'model': 'm1', 'epochs': '3', 'batch': '4', 'user_id': 'u1'}
    files = {'dataset': ('data.zip', b'PK\x03\x04', 'application/zip')}
    r = client.post("/train_yolo", data=data, files=files)
    assert r.status_code == 200
    json_resp = r.json()
    assert 'job_id' in json_resp
