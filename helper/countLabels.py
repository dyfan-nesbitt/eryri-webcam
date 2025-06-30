import json
from collections import Counter

# Load labels
with open('balanced_output_binary_vectors_no_snow.json', 'r') as f:
    labels_dict = json.load(f)

# Count labels
label_counts = Counter([tuple(v) for v in labels_dict.values()])

# Print label
print("Label occurences:")
for label, count in label_counts.items():
    print(f"{label}: {count}")
