import json

# Label map that defines the indices of each label
label_map = {
    "Summit Visible": 0,
    "Clear Sky": 1
}

# Converts labels to binary vector
def labels_to_vector(labels, label_map):
    # Initialize a vector with zeros
    vector = [0] * len(label_map)
    for label in labels:
        if label in label_map:
            vector[label_map[label]] = 1
    return vector

# Processes the input JSON and output binary vectors
def process_labels(input_json_path, output_json_path):
    # Read the input JSON file
    with open(input_json_path, "r") as f:
        data = json.load(f)
    
    # Holds the image paths and their binary label vectors
    image_vectors = {}
    
    # Iterate through the images and their labels
    for image, labels in data.items():
        # Convert the labels to a binary vector
        binary_vector = labels_to_vector(labels, label_map)
        # Store the binary vector in the dictionary
        image_vectors[image] = binary_vector
    
    # Write the result to the output JSON file
    with open(output_json_path, "w") as f:
        json.dump(image_vectors, f, indent=4)

    print(f"Processed {len(image_vectors)} images and saved the binary vectors to {output_json_path}.")

# Example usage
input_json_path = "labelsNoSnow.json"  # Input JSON file path
output_json_path = "output_binary_vectors_no_snow.json"  # Output JSON file path

process_labels(input_json_path, output_json_path)
