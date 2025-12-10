import os
from ultralytics.models import YOLO
import shutil

def train(directory: str, dataset: str, epochs: int = 100, batch: int = 16):
    os.makedirs(directory, exist_ok=True)
    
    model = YOLO("yolov8n.pt")

    model_results = model.train(
        data=os.path.join(dataset, 'dataset', 'data.yaml') if os.path.exists(os.path.join(dataset, 'dataset', 'data.yaml')) else os.path.join(dataset, 'dataset'),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        project=directory,
        name='training_run',
        exist_ok=True
    )

    trained_weights_path = os.path.join(directory, 'training_run', 'weights', 'best.pt')
    destination_path = os.path.join(directory, 'final_model.pt')

    if os.path.exists(trained_weights_path):
        shutil.move(trained_weights_path, destination_path)
        print(f"Final weights moved to: {destination_path}")
        try:
            shutil.rmtree(os.path.join(directory, 'training_run'))
        except Exception:
            pass

        try:
            y = YOLO(destination_path)
            onnx_path = os.path.join(directory, 'final_model.onnx')
            y.export(format='onnx', imgsz=640, opset=12, save=True)

            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                quant_path = os.path.join(directory, 'final_model.quant.onnx')
                quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
                print(f"Quantized model saved to: {quant_path}")
            except Exception as e:
                print(f"ONNX quantization skipped: {e}")
        except Exception as e:
            print(f"ONNX export/quantization failed: {e}")

    return model_results