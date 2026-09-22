from ultralytics import YOLO

# Load the severity model once upon module import to optimize memory and latency
print("Loading severity classification model...")
severity_model = YOLO("best_severity.pt")

def classify_fire_crop(fire_crop):
    """
    Takes an OpenCV image crop of a detected fire region and returns its severity tier and confidence score.
    """
    if fire_crop is None or fire_crop.size == 0:
        return "Unknown", 0.0
        
    # Run classification inference on the region of interest
    results = severity_model(fire_crop, verbose=False)
    
    # Extract prediction results
    top_class_idx = results[0].probs.top1
    confidence = results[0].probs.top1conf.item()
    label = severity_model.names[top_class_idx]
    
    return label, confidence