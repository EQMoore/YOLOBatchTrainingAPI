import argparse
import tempfile
import trainer_gcs_util
import zipfile
import train
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_zip', type=str, required=True, help='GCS path to the dataset zip file (gs://bucket/path.zip)')
    parser.add_argument('--model', type=str, required=True, help='Model name (used for output blob path)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=32, help='Batch size for training')

    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        temp_zip_path = temp_zip.name

    try:
        trainer_gcs_util.download_from_gcs(temp_zip_path, args.dataset_zip)

        with tempfile.TemporaryDirectory() as extract_dir:
            try:
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                with tempfile.TemporaryDirectory() as work_dir:
                    train_results = train.train(work_dir, extract_dir)

                    final_pt = os.path.join(work_dir, 'final_model.pt')
                    if os.path.exists(final_pt):
                        out_blob = f"{args.model}/final_model.pt"
                        trainer_gcs_util.upload_model(final_pt, f"gs://{os.getenv('BUCKET_NAME')}/{out_blob}")

                    quant_onx = os.path.join(work_dir, 'final_model.quant.onnx')
                    if os.path.exists(quant_onx):
                        out_blob = f"{args.model}/final_model.quant.onnx"
                        trainer_gcs_util.upload_model(quant_onx, f"gs://{os.getenv('BUCKET_NAME')}/{out_blob}")

            except zipfile.BadZipFile:
                print(f"Error: {temp_zip_path} is not a valid ZIP file.")
    finally:
        try:
            os.remove(temp_zip_path)
        except Exception:
            pass


if __name__ == '__main__':
    main()