import os
import cv2
import shutil

# Paths for the image folders
IMAGE_FOLDER = "static/images"
DARK_FOLDER = "static/dark_images"
THRESHOLD = 100  # Brightness threshold

image_num = 0

# Make sure the dark images folder exists
os.makedirs(DARK_FOLDER, exist_ok=True)

def is_dark(image_path, threshold=THRESHOLD):#
    """Return True if the image is too dark."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        print(f"Warning: Unable to read {image_path}")
        return False
    mean_brightness = img.mean()  # Calculate average brightness
    return mean_brightness < threshold

def filter_dark_images():
    """Move dark images from the IMAGE_FOLDER to the DARK_FOLDER."""
    dark_images = []
    total_images = 0

    for img_name in os.listdir(IMAGE_FOLDER):
        img_path = os.path.join(IMAGE_FOLDER, img_name)

        total_images += 1  # Increment total image counter
        print(f"Images checked: {total_images}", end="\r")

        if is_dark(img_path):
            dark_images.append(img_name)
            dest_path = os.path.join(DARK_FOLDER, img_name)
            shutil.move(img_path, dest_path)  # Move the image
            print(f"Moved dark image: {img_name}")
    
    print(f"Total images checked: {total_images}")
    print(f"Total dark images moved: {len(dark_images)}")

if __name__ == "__main__":
    filter_dark_images()
