import os
import cv2

# Paths to your current detection dataset
IMAGE_DIR = "dataset/train/images"
LABEL_DIR = "dataset/train/labels"

# Output folder for your new classifier dataset
OUTPUT_DIR = "dataset_severity/train"

# Create severity subfolders if they don't exist
for tier in ["low", "moderate", "high", "critical"]:
    os.makedirs(os.path.join(OUTPUT_DIR, tier), exist_ok=True)

# Loop through all images in your training directory
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.endswith((".jpg", ".png", ".jpeg")):
        continue
        
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))
    
    if not os.path.exists(label_path):
        continue
        
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        class_id = int(parts[0])
        
        # Assuming class 0 is "fire" (adjust if your class map is different)
        if class_id == 0: 
            # YOLO format is normalized: [x_center, y_center, width, height]
            x_center, y_center, box_w, box_h = map(float, parts[1:])
            
            # Convert to pixel coordinates
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)
            
            # Crop the fire region from the image
            fire_crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            if fire_crop.size > 0:
                # Save crop into a temporary folder where you can sort them
                output_path = os.path.join(OUTPUT_DIR, "low", f"{img_name}_{idx}.jpg")
                cv2.imwrite(output_path, fire_crop)

print("Extraction complete! Check dataset_severity/train/low to sort your crops.")