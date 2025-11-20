spec:
import os
import uuid
from googleapiclient import discovery

PROJECT = os.getenv("PROJECT_ID")
ZONE = os.getenv("ZONE", "us-central1-a")
BUCKET = os.getenv("BUCKET_NAME")
CONTAINER_IMAGE = os.getenv("VERTEXT_CONTAINER_URI")


def create_training_vm(dataset_gcs_path: str, model_name: str, epochs: int = 10, batch: int = 16, machine_type: str = "n1-standard-8") -> str:
    if not PROJECT:
        raise RuntimeError("PROJECT_ID environment variable not set")
    if not BUCKET:
        raise RuntimeError("BUCKET_NAME environment variable not set")
    if not CONTAINER_IMAGE:
        raise RuntimeError("VERTEXT_CONTAINER_URI environment variable not set")

    compute = discovery.build('compute', 'v1')

    instance_name = f"yolo-trainer-{uuid.uuid4().hex[:8]}"

    container_declaration = f"""
spec:
  containers:
    - name: trainer
      image: {CONTAINER_IMAGE}
      env:
        - name: BUCKET_NAME
          value: "{BUCKET}"
      args: ['--dataset_zip={dataset_gcs_path}','--model={model_name}','--epochs={epochs}','--batch={batch}']
  restartPolicy: Never
"""

    machine_type_full = f"zones/{ZONE}/machineTypes/{machine_type}"

    config = {
        'name': instance_name,
        'machineType': machine_type_full,
        'disks': [
            {
                'boot': True,
                'autoDelete': True,
                'initializeParams': {
                    'sourceImage': 'projects/cos-cloud/global/images/family/cos-stable'
                }
            }
        ],
        'networkInterfaces': [
            {
                'network': 'global/networks/default',
                'accessConfigs': [ {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'} ]
            }
        ],
        'serviceAccounts': [
            {
                'email': 'default',
                'scopes': [
                    'https://www.googleapis.com/auth/cloud-platform'
                ]
            }
        ],
        'metadata': {
            'items': [
                {
                    'key': 'gce-container-declaration',
                    'value': container_declaration
                }
            ]
        }
    }

    request = compute.instances().insert(project=PROJECT, zone=ZONE, body=config)
    response = request.execute()

    op_name = response.get('name')
    return op_name
