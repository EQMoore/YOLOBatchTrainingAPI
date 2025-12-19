import argparse
import tempfile
import trainer_gcs_util
import zipfile
import train
import os

def main():
    #handle parser for CLI input
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_zip', type=str, required=True, help='GCS path to the dataset zip file (gs://bucket/path.zip)')
    parser.add_argument('--model', type=str, required=True, help='Model name (used for output blob path)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=32, help='Batch size for training')

    '''
    In all serverless training jobs, Vertex AI mounts Cloud Storage buckets that you have access to bin the /gcs/ 
    directory of each training node's file system. As a convenient alternative to using the Python Client for Cloud 
    Storage or another library to access Cloud Storage, you can read and write directly to the local file system in 
    order to read data from Cloud Storage or write data to Cloud Storage. Job might not need to upload back to gcs. 
    Vertex is also integrated with google cloud ai hosting so I might save this container as a users cloud model.
    '''
    args = parser.parse_args()

    #tempfile will not delete when it closes (delete=false)
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        temp_zip_path = temp_zip.name
    try:
        #try downloading from gcs into this tempfile
        trainer_gcs_util.download_from_gcs(temp_zip_path, args.dataset_zip)
        
        #create a new temporary directory to store the extracted zip file
        with tempfile.TemporaryDirectory() as extract_dir:
            try:
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                #because delete != false in the tempDir, it should delete after use in the 'with' block
                with tempfile.TemporaryDirectory() as work_dir:
                    train_results = train.train(work_dir, extract_dir)

                    #train the model in this temporary directory
                    final_pt = os.path.join(work_dir, 'final_model.pt')
                    if os.path.exists(final_pt):
                        out_blob = f"{args.model}/final_model.pt"
                        #upload the final unquantized model to gcs
                        trainer_gcs_util.upload_model(final_pt)
                    
                    #get the quantized model
                    quant_onx = os.path.join(work_dir, 'final_model.quant.onnx')
                    if os.path.exists(quant_onx):
                        out_blob = f"{args.model}/final_model.quant.onnx"
                        #upload the quantized model to gcs
                        trainer_gcs_util.upload_model(quant_onx)
                        
            #handle an error in the case that the program can not extract the zip file
            except zipfile.BadZipFile:
                print(f"Error: {temp_zip_path} is not a valid ZIP file.")
    finally:
        try:
            #delete the temp_zip_path because delete != true
            os.remove(temp_zip_path)
        except Exception:
            pass


if __name__ == '__main__':
    main()