from ultralytics import YOLO

# 1. Load your trained weights
model = YOLO("best.pt")

# 2. Run inference and automatically save the annotated output video
results = model.predict(
    source="test_videos/none.mp4",
    imgsz=1024,
    conf=0.05,
    save=True,               # Automatically saves the annotated video
    project="runs/detect",   # Saves output folder inside runs/detect/
    name="drone_test"        # Output subfolder name
)

# 3. Process detections frame by frame
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]  # 'fire' or 'smoke'
        print(f"Detected {class_name} with confidence {conf:.2f}")
        