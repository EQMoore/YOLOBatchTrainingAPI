spec:
import os
import uuid
from googleapiclient import discovery

PROJECT = os.getenv("PROJECT_ID")
ZONE = os.getenv("ZONE", "us-central1-a")
BUCKET = os.getenv("BUCKET_NAME")
CONTAINER_IMAGE = os.getenv("VERTEXT_CONTAINER_URI")


def create_training_vm(dataset_gcs_path: str, model_name: str, epochs: int = 10, batch: int = 16, machine_type: str = "n1-standard-8") -> dict:
    if not PROJECT:
        raise RuntimeError("PROJECT_ID environment variable not set")
    if not BUCKET:
        raise RuntimeError("BUCKET_NAME environment variable not set")
    if not CONTAINER_IMAGE:
        raise RuntimeError("VERTEXT_CONTAINER_URI environment variable not set")

    compute = discovery.build('compute', 'v1')

    instance_name = f"yolo-trainer-{uuid.uuid4().hex[:8]}"

    container_declaration = (
        "spec:\n"
        "  containers:\n"
        "    - name: trainer\n"
        f"      image: {CONTAINER_IMAGE}\n"
        "      env:\n"
        "        - name: BUCKET_NAME\n"
        f"          value: \"{BUCKET}\"\n"
        f"      args: ['--dataset_zip={dataset_gcs_path}','--model={model_name}','--epochs={epochs}','--batch={batch}']\n"
        "  restartPolicy: Never\n"
    )

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
    return {"operation": op_name, "instance": instance_name}


def get_zone_operation_status(operation_name: str) -> dict:
    compute = discovery.build('compute', 'v1')
    return compute.zoneOperations().get(project=PROJECT, zone=ZONE, operation=operation_name).execute()


def get_instance_status(instance_name: str) -> dict:
    compute = discovery.build('compute', 'v1')
    return compute.instances().get(project=PROJECT, zone=ZONE, instance=instance_name).execute()


def get_serial_port_output(instance_name: str, port: int = 1) -> dict:
    compute = discovery.build('compute', 'v1')
    return compute.instances().getSerialPortOutput(project=PROJECT, zone=ZONE, instance=instance_name, port=port).execute()


def delete_instance(instance_name: str) -> dict:
    compute = discovery.build('compute', 'v1')
    return compute.instances().delete(project=PROJECT, zone=ZONE, instance=instance_name).execute()


def wait_for_instance_and_delete(instance_name: str, poll_interval: int = 15, timeout: int = 3600) -> dict:
    import time
    compute = discovery.build('compute', 'v1')
    start = time.time()
    while True:
        inst = compute.instances().get(project=PROJECT, zone=ZONE, instance=instance_name).execute()
        status = inst.get('status')
        if status and status.upper() != 'RUNNING':
            return compute.instances().delete(project=PROJECT, zone=ZONE, instance=instance_name).execute()
        if time.time() - start > timeout:
            raise TimeoutError('timeout waiting for instance state change')
        time.sleep(poll_interval)
