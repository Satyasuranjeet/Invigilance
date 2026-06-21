import os
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)

def main() -> None:
    """Downloads YOLOv8n weights to the models/ directory."""
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "yolov8n.pt")
    
    logging.info(f"Downloading/loading YOLOv8n model to {model_path}...")
    model = YOLO("yolov8n.pt") # downloads to current dir if not exists
    
    # move to models dir if it was downloaded locally
    if os.path.exists("yolov8n.pt"):
        os.rename("yolov8n.pt", model_path)
    
    # Reload from models/ to confirm it works
    model = YOLO(model_path)
    logging.info(f"Model loaded successfully from {model_path}.")

if __name__ == "__main__":
    main()
