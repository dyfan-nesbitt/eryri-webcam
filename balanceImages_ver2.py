import os
import json
import shutil
import random
from collections import Counter

# Load labels
with open('output_binary_vectors_no_snow.json', 'r') as f:
    labels_dict = json.load(f)

# Define paths
original_image_dir = 'static/images_labeled'
balanced_image_dir = 'static/images_balanced'
os.makedirs(balanced_image_dir, exist_ok=True)

# Count labels
label_counts = Counter([tuple(v) for v in labels_dict.values()])

# Sort labels by their count
sorted_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

# Find the second most frequent label
label_to_balance = sorted_labels[1][0]

# Print label occurrences
print("Label occurrences:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# Prepare to balance the dataset
new_labels_dict = {}

# Create a list of image files for each label
images_per_label = {label: [img for img, lbl in labels_dict.items() if tuple(lbl) == label] for label in label_counts}

# Use the second most label's count as the reference for trimming images from the other label
for label, files in images_per_label.items():
    if label != label_to_balance:
        random.shuffle(files)
        # Trim down the list to match the count of the second most label
        files = files[:label_counts[label_to_balance]]
    
    # Copy images to the balanced directory
    for img_file in files:
        src = os.path.join(original_image_dir, img_file)
        dst = os.path.join(balanced_image_dir, img_file)
        shutil.copy2(src, dst)
        new_labels_dict[img_file] = list(label)

# Save the new balanced labels
with open("balanced_output_binary_vectors_no_snow.json", "w") as f:
    json.dump(new_labels_dict, f, indent=4)

print("Balanced dataset created!")
