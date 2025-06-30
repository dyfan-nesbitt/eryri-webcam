from flask import Flask, render_template, request, jsonify, send_from_directory, abort
import os
import json
import tensorflow as tf

app = Flask(__name__)

# Set folders and file paths
IMAGE_FOLDER = "static/images"
LABELS_FILE = "labelsNoSnow.json"

# Ensure IMAGE_FOLDER exists
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Load labels safely
try:
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            labels = json.load(f)
    else:
        labels = {}
except (json.JSONDecodeError, IOError):
    print("Warning: labels file is corrupted. Starting fresh.")
    labels = {}

model = tf.keras.models.load_model('model_test.keras', compile=False)
# Define the label order 
LABEL_NAMES = ["Summit Visible", "Clear Sky"]

def predict_labels(image_path, threshold=0.7):
    """
    Loads an image, preprocesses it, and predicts its labels using the model.
    Returns a list of predicted labels based on the threshold.
    """
    # Read and decode the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [400, 400])
    # Normalize the image to [0, 1]
    image = image / 255.0
    # Add batch dimension
    image = tf.expand_dims(image, axis=0)
    
    # Get predictions from the model
    preds = model.predict(image)
    preds = preds[0]  # remove batch dimension
    
    # Determine predicted labels based on the threshold
    predicted_labels = []
    for i, score in enumerate(preds):
        if score >= threshold:
            predicted_labels.append(LABEL_NAMES[i])
    return predicted_labels

@app.route("/")
def index():
    # List only image files
    images = sorted(
        [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    
    if not images:
        print("Warning: No images found in static/images/")
    
    for img in images:
        if img not in labels:
            img_path = os.path.join(IMAGE_FOLDER, img)
            labels[img] = predict_labels(img_path)
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=4)
    
    return render_template("predict.html", images=images, labels=labels)

@app.route("/images/<filename>")
def get_image(filename):
    """Serve images safely"""
    img_path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(img_path):
        abort(404)  # Return 404 if image not found
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route("/predict_labels/<filename>")
def predict_labels_route(filename):
    """Return predicted labels for a given image as JSON."""
    img_path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(img_path):
        return jsonify({"error": "Image not found"}), 404
    predicted = predict_labels(img_path)
    return jsonify({"predicted_labels": predicted})

@app.route("/save_labels", methods=["POST"])
def save_labels():
    try:
        data = request.json
        if not data or "image" not in data or "labels" not in data:
            return jsonify({"error": "Invalid request data"}), 400
        
        image_name = data["image"]
        image_labels = data["labels"]
        
        labels[image_name] = image_labels
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f, indent=4)

        return jsonify({"message": "Labels saved successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
