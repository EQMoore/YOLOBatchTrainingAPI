import argparse
import tempfile
import trainer_gcs_util
import zipfile
import train
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_zip', type=str, required=True, help='GCS path to the dataset zip file')
    parser.add_argument('--blob', type=str, required=True, help='Model name or path')
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    with tempfile.TemporaryFile(mode='wb', encoding='utf-8') as temp_zip:
        trainer_gcs_util.download_from_gcs(temp_zip.name, args.__getattribute__('--model'))
        try:
            with zipfile.ZipFile(temp_zip.name, 'r') as zip_ref:
                with tempfile.TemporaryFile(mode='wb', encoding='utf-8') as temp_file:
                    zip_ref.extractall(temp_file)
                    train.train(temp_file.name, os.path.join(temp_file.name, 'model'))
                    trainer_gcs_util.upload_model(os.path.join(temp_file.name, 'model'))
        except FileNotFoundError:
            print(f"Error: The file {temp_zip.name} was not found.")
        except zipfile.BadZipFile:
            print(f"Error: {temp_zip.name} is not a valid ZIP file.")

if __name__ == '__main__':
    main()