import os
import shutil
import json

# Path to your JSON file and the folder to move images
json_file_path = 'labelsNoSnow.json'
source_directory = 'static/images'
destination_directory = 'static/images_labeled'

# Ensure the destination folder exists
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Loop through the JSON data and move the files
for image_file, labels in data.items():
    # Build full path to the image file
    source_path = os.path.join(source_directory, image_file)
    
    # Check if the file exists
    if os.path.exists(source_path):
        destination_path = os.path.join(destination_directory, image_file)
        
        # Move the image to the labeled_images folder
        shutil.move(source_path, destination_path)
        print(f'Moved: {image_file}')
    else:
        print(f'File not found: {image_file}')
