from flask import Flask, render_template, request, jsonify, send_from_directory, abort
import os
import json

app = Flask(__name__)

IMAGE_FOLDER = "static/images"
LABELS_FILE = "labels.json"

# Ensure IMAGE_FOLDER exists
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
    print("No image directory found.")

# Load labels safely
try:
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            labels = json.load(f)
    else:
        labels = {}
except (json.JSONDecodeError, IOError):
    print("Warning: labels.json is corrupted. Starting fresh.")
    labels = {}

@app.route("/")
def index():
    # Filter only image files
    images = sorted(
        [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    
    if not images:
        print("Warning: No images found in static/images/")
    
    return render_template("index.html", images=images, labels=labels)

@app.route("/images/<filename>")
def get_image(filename):
    """Serve images safely"""
    img_path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(img_path):
        abort(404)  # Return 404 if image not found
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route("/save_labels", methods=["POST"])
def save_labels():
    try:
        data = request.json
        if not data or "image" not in data or "labels" not in data:
            return jsonify({"error": "Invalid request data"}), 400  # Return an error
        
        image_name = data["image"]
        image_labels = data["labels"]
        
        labels[image_name] = image_labels
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f, indent=4)

        return jsonify({"message": "Labels saved successfully!"})  # Correct JSON response
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Catch unexpected errors

if __name__ == "__main__":  
    app.run(debug=True)
