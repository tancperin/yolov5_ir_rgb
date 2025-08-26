import os
from yolov5 import train

# Paths
DATA_YAML = "dataset/data.yaml"

# Training parameters
epochs = 10  # You can increase this for better results
imgsz = 640
batch_size = 2
weights = 'yolov5s.pt'  # Use a small model for quick test

if __name__ == '__main__':
    train.run(
        data=DATA_YAML,
        imgsz=imgsz,
        batch_size=batch_size,
        epochs=epochs,
        weights=weights,
        project='runs/train',
        name='rgb_001',
        exist_ok=True
    )
