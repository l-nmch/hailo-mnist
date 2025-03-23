import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

import argparse

# use argparse to get the image folder
parser = argparse.ArgumentParser(description='Create a calibration set for the model.')
parser.add_argument('--image_folder', type=str, required=True, help='Name of the folder containing the images')
parser.add_argument('--output', type=str, required=True, help='Name of the output file .npy')
args = parser.parse_args()

# Paramètres
image_folder = Path(args.image_folder)
img_height = 28
img_width = 28
img_channels = 1

image_paths = list(image_folder.glob("*.png"))

calib_size = len(image_paths)
print(f"Nombre d'images trouvées : {calib_size}")

calib_set = np.zeros((calib_size, img_height, img_width, img_channels), dtype=np.float32)

for i, img_path in enumerate(image_paths):
    image = Image.open(img_path)
    img_resized = image.resize((img_width, img_height))
    img_inverted = ImageOps.invert(img_resized.convert("L"))
    img_array = np.array(img_inverted)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (224, 224, 1)
    img_array = np.repeat(img_array, img_channels, axis=-1)  # (224, 224, 3)
    calib_set[i] = img_array

np.save(args.output, calib_set)
print(calib_set)
print("Calibration set créé avec succès !")
