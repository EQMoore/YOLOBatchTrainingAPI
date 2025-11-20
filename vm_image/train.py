import os
from ultralytics import YOLO
import shutil

def train(directory: str, dataset: str):
    os.makedirs(directory, exist_ok=True)
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=os.path.join(dataset, 'dataset', 'train'),
        epochs=100,
        imgsz=640,
        batch=16,
        project=directory,
        name='training_run',
        exist_ok=True
    )
    
    final_weights_path = os.path.join(dataset, 'training_run', 'weights', 'best.pt')
    destination_path = os.path.join(directory, 'final_model.pt')
    
    if os.path.exists(final_weights_path):
        shutil.move(final_weights_path, destination_path)
        print(f"Final weights moved to: {destination_path}")
        shutil.rmtree(os.path.join(directory, 'training_run'))
        
    return results