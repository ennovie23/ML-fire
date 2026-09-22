import os
from datasets import load_dataset

print("Loading dataset from cache...")
dataset = load_dataset("AbdullahImran/DeepLearningProject")
class_names = dataset["train"].features["label"].names  # ['mild', 'moderate', 'severe']

# Loop through all splits (train, validation, test)
for split in dataset.keys():
    print(f"Exporting {split} split...")
    split_data = dataset[split]
    
    for idx, item in enumerate(split_data):
        img = item["image"]
        label_idx = item["label"]
        label_name = class_names[label_idx]
        
        # Define destination directory: dataset_severity/train/mild/, etc.
        out_dir = os.path.join("dataset_severity", split, label_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # Save the image
        img_path = os.path.join(out_dir, f"{split}_{idx}.jpg")
        img.save(img_path)

print("Export complete! Your dataset is now ready in the 'dataset_severity' folder.")