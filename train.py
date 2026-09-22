from ultralytics import YOLO

def main():
    model = YOLO("yolov10n.pt")

    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,           # Safe batch size for M1 memory footprint
        val=False,          # Bypasses per-epoch validation loop
        workers=2,          # Parallel data loading
        name="fire_smoke_yolov10",
        device="mps"
    )

if __name__ == "__main__":
    main()


    from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 classification model (-cls)
    # y8n-cls is lightweight and fast for secondary classification heads
    model = YOLO("yolov8n-cls.pt")

    # Train the model on your severity dataset
    results = model.train(
        data="dataset_severity",  # Path to your folder
        epochs=30,                # Number of training iterations
        imgsz=224,                # Standard image size for classification
        batch=16,                 # Batch size (adjust if you run into memory limits)
        device="cpu"              # Change to 0 if you are using an NVIDIA GPU, or leave as 'cpu'/'mps' for Mac
    )

    print("Training complete! Model saved to runs/classify/train/weights/best.pt")

if __name__ == "__main__":
    main()