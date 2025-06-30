import os
import json
import tensorflow as tf
import numpy as np
import cv2
import random
import shutil
import tweepy
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Load model
model = tf.keras.models.load_model('model_test.keras')

CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    else:
        # Default thresholds
        config = {
            "social_media": {"summit": 0.8, "clear_sky": 0.5},
            "social_media_clear": {"summit": 0.9, "clear_sky": 0.8},
            "good_image": {"summit": 0.5}
        }
    return config

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

config = load_config()

def preprocess_image(image_path, target_size=(400, 400)):
    # Load and preprocess image for prediction
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, target_size)
    image = image / 255.0 # Normalize
    # Add batch dimension 
    image = tf.expand_dims(image, axis=0) # shape is now (1, 400, 400, 3)
    return image

def generate_social_media_post(prediction):
    # Generate post if the image qualifies
    summit_prob = prediction[0][0]
    clear_sky_prob = prediction[0][1]

    if summit_prob > config["social_media"]["summit"] and clear_sky_prob > config["social_media"]["clear_sky"]:
        # If the sky is definitely clear then include in the textual description
        if summit_prob > config["social_media_clear"]["summit"] and clear_sky_prob > config["social_media_clear"]["clear_sky"]:
            templates = [
                "Stunning view of Snowdon! Summit visible and clear skies. Nature at its best! #Snowdon #SummitView",
                "What an incredible day on Snowdon with summit visibility and  clear skies. Loving the outdoors! #Mountains",
                "Enjoying the perfect blend of nature on Snowdon — summit visible and clear skies. Simply breathtaking! #ScenicViews"
            ]
            # Randomly choose between the three strings
            template = random.choice(templates)
            return template.format(s_prob=summit_prob*100, c_prob=clear_sky_prob*100)
        # Otherwise just mention summit visible
        templates = [
            "Stunning view of Snowdon! Summit visible. Nature at its best! #Snowdon #SummitView",
            "What an incredible day on Snowdon with summit visibility. Loving the outdoors! #MountainMagic",
            "Enjoying the perfect blend of nature on Snowdon — summit visible. Simply breathtaking! #ScenicViews"
        ]
        # Randomly choose between the three strings
        template = random.choice(templates)
        return template.format(s_prob=summit_prob*100, c_prob=clear_sky_prob*100)
    return None

def is_good_for_archive(prediction):
    summit_prob = prediction[0][0]
    # Check if the image is good enough to archive
    if summit_prob > config['good_image']['summit']:
        return True
    return False

def archive_image(image_path, archive_dir='static/archive'):
    # Archive image
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    name = os.path.basename(image_path)
    archive_path = os.path.join(archive_dir, name)
    shutil.copy2(image_path, archive_path)
    return archive_path

def upload_to_google_drive(file_path, folder_id=None):
    try: 
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
    except ImportError:
        update_output("PyDrive2 is not installed. Please install it to enable Google Drive uploading.\n")
        return False, "PyDrive2 not installed\n"

    try:
        gauth = GoogleAuth()
        # Try to load saved creds, if they don't exist run local webserver authentication
        if os.path.exists("mycreds.txt"):
            gauth.LoadCredentialsFile("mycreds.txt")
        else:
            gauth.LocalWebserverAuth() # Opens browser for auth
            gauth.SaveCredentialsFile("mycreds.txt")
        drive = GoogleDrive(gauth)

        file_name = os.path.basename(file_path)
        metadata = {'title': file_name}
        if folder_id: # Get the folder that the file should be stored in i.e "archive"
            metadata['parents'] = [{'id': folder_id}]
        drive_file = drive.CreateFile(metadata)
        drive_file.SetContentFile(file_path)
        drive_file.Upload()
        return True, drive_file['id'] + "\n"
    except Exception as e:
        return False, str(e)
    
def post_to_social_media(description, image_path):
    # Placeholder function
    api_key = os.getenv("SOCIAL_MEDIA_API_KEY")
    if not api_key:
        update_output("No API key found. Skipping social media post.\n")
        return
    # Future impl: Integrate with chosen social media
    update_output("Posting to social media...\n")
    update_output(f"Description: {description}")
    update_output(f"Image path: {image_path}")
    # Posting code
    return

def update_output(message):
    # Helper function to add a message to the output.
    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, message + "\n")
    output_text.see(tk.END)
    output_text.config(state=tk.DISABLED)

def process_image(image_path):
    update_output("-----Image Processing Starting-----\n")
    update_output(f"Selected image: {image_path}\n")
    # Preprocess and predict image
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    summit_prob = round(float(prediction[0][0]) * 100, 1)
    clear_sky_prob = round(float(prediction[0][1]) * 100, 1)
    predictions_label.config(text=f"Summit Visibility: {summit_prob}%\n Clear Sky: {clear_sky_prob}%")
    update_output(f"Summit Visibility: {summit_prob}%, Clear Sky: {clear_sky_prob}%\n")
    
    # Generate social media text if the images are qualifies
    description = generate_social_media_post(prediction)
    if description:
        update_output("Image qualifies for a social media post. Generated post:\n")
        update_output(description + "\n")
        archived_path = archive_image(image_path)
        update_output(f"Image archived at: {archived_path}\n")
        google_archive_folder = "1jqm6wbDalRyBRYJem_MPaf94p7m1BFc2"
        success, result = upload_to_google_drive(archived_path, google_archive_folder)
        if success:
            update_output(f"Successfully uploaded to Google Drive. File ID: {result}\n")
        else:
            update_output(f"Google Drive upload failed: {result}\n")
        post_to_social_media(description, archived_path)
    else:
        if is_good_for_archive(prediction):
            archived_path = archive_image(image_path)
            update_output(f"Image qualifies as a good image and was archived at: {archived_path}\n")
            google_archive_folder = "1jqm6wbDalRyBRYJem_MPaf94p7m1BFc2"
            success, result = upload_to_google_drive(archived_path, google_archive_folder)
            if success:
                update_output(f"Successfully uploaded to Google Drive. File ID: {result}\n")
            else:
                update_output(f"Google Drive upload failed: {result}\n")
        else:
            update_output("Image does not meet the thresholds for archiving.\n")

    update_output("-----Image Processing Complete-----\n")

def select_image():
    # Select the image from the file system
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "jpg")]
    )
    if file_path:
        pil_image = Image.open(file_path)
        pil_image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(pil_image)
        image_label.config(image=photo)
        image_label.image = photo
        process_image(file_path)

def open_settings_window():
    # Open settings panel where the user can adjust thresholds
    settings_win = tk.Toplevel(root)
    settings_win.title("Adjust Settings")
    settings_win.geometry("400x450")

    # Create variables for sliders
    sm_summit = tk.DoubleVar(value=config["social_media"]["summit"])
    sm_clear = tk.DoubleVar(value=config["social_media"]["clear_sky"])
    smc_summit = tk.DoubleVar(value=config["social_media_clear"]["summit"])
    smc_clear = tk.DoubleVar(value=config["social_media_clear"]["clear_sky"])
    gi_summit = tk.DoubleVar(value=config["good_image"]["summit"])

    # Sliders
    tk.Label(settings_win, text="Social Media - Summit Threshold").pack(pady=5)
    tk.Scale(settings_win, variable=sm_summit, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)

    tk.Label(settings_win, text="Social Media - Clear Sky Threshold").pack(pady=5)
    tk.Scale(settings_win, variable=sm_clear, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)

    tk.Label(settings_win, text="Social Media w/ Clear - Summit Threshold").pack(pady=5)
    tk.Scale(settings_win, variable=smc_summit, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)

    tk.Label(settings_win, text="Social Media w/ Clear - Clear Sky Threshold").pack(pady=5)
    tk.Scale(settings_win, variable=smc_clear, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
    
    tk.Label(settings_win, text="Archive Image - Summit Threshold").pack(pady=5)
    tk.Scale(settings_win, variable=gi_summit, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)

    def save_settings():
        config["social_media"]["summit"] = sm_summit.get()
        config["social_media"]["clear_sky"] = sm_clear.get()
        config["social_media_clear"]["summit"] = smc_summit.get()
        config["social_media_clear"]["clear_sky"] = smc_clear.get()
        config["good_image"]["summit"] = gi_summit.get()

        save_config(config)
        update_output("Settings updated and saved.")
        settings_win.destroy()

    tk.Button(settings_win, text="Save Settings", command=save_settings, font=("Arial", 14)).pack(pady=20)

# Create TKinter root
root = tk.Tk()
root.title("Snowdon Image Processor")
root.geometry("800x600")

# Menu bar
menu_bar = tk.Menu(root)
settings_menu = tk.Menu(menu_bar, tearoff=0)
settings_menu.add_command(label="Adjust Thresholds", command=open_settings_window)
menu_bar.add_cascade(label="Settings", menu=settings_menu)
root.config(menu=menu_bar)

# Create top frame - explanation, button and image preview
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# In the top frame, create left and right frame
top_left_frame = tk.Frame(top_frame)
top_left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

top_right_frame = tk.Frame(top_frame)
top_right_frame.grid(row=0, column=1, sticky="nsew", padx= 10, pady=10)

# Grid weight for equal expansion
top_frame.columnconfigure(0, weight=1)
top_frame.columnconfigure(1, weight=1)
top_frame.rowconfigure(0, weight=1)

explanation = (
    "Welcome to the Snowdon Image Processor!\n\n"
    "This application allows you to select an image of Snowdon and automatically processes it to predict "
    "the visibility of the summit and the clarity of the sky. If the image meets high thresholds, a social "
    "media post description is generated and the image is archived. Even if it doesn't qualify for a post, "
    "good images are still archived for future reference.\n"
)
explanation_label = tk.Label(top_left_frame, text=explanation, justify=tk.LEFT, wraplength=500, font=("Arial", 16, "bold"))
explanation_label.pack(padx=10, pady=10)

# Select Button
select_button = tk.Button(top_left_frame, text="Select Image", command=select_image, font=("Arial", 18, "bold"), padx=15, pady=5)
select_button.pack(pady=10)

# Preview Image
image_label = tk.Label(top_right_frame)
image_label.pack(anchor="n", padx=10, pady=10)

predictions_label = tk.Label(top_right_frame, text="", font=("Ariel", 14))
predictions_label.pack(pady=5)

# Create bottom frame for output
bottom_frame = tk.Frame(root)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

output_text = tk.Text(bottom_frame, height=10, state=tk.DISABLED, wrap=tk.WORD, font=(12))
output_text.pack(fill=tk.BOTH, expand=True)

root.mainloop()