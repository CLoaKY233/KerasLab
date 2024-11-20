# File: data_loader.py
# Purpose: Load and preprocess image data

# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Developer details:
#   Name         : Lay Sheth and Rishav Raj
#   Role         : Software Engineers
#   Version      : V 1.0
#   Unit test    : Pass
#   Integration test: Pass
#   Description  : This script loads and preprocesses image data from the specified directory.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import cv2
import numpy as np
from pyfiles.constants import IMG_SIZE

def load_data(path):
    """
    Load and preprocess image data from the specified directory.

    Args:
        path (str): Path to the dataset directory.

    Returns:
        tuple: Tuple containing two numpy arrays:
               - x (numpy.ndarray): Array of processed images.
               - y (numpy.ndarray): Array of corresponding labels (0 for dogs, 1 for cats).
    """
    x_dataset = []
    y_dataset = []

    # Load cat images
    cats_path = os.path.join(path, "cats")
    for img in os.listdir(cats_path):
        process_image(os.path.join(cats_path, img), x_dataset, y_dataset, 1)

    # Load dog images
    dogs_path = os.path.join(path, "dogs")
    for img in os.listdir(dogs_path):
        process_image(os.path.join(dogs_path, img), x_dataset, y_dataset, 0)

    x = np.array(x_dataset)
    y = np.array(y_dataset)
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    return x[indices], y[indices]

def process_image(img_path, x_dataset, y_dataset, label):
    """
    Process a single image and append it to the dataset.

    Args:
        img_path (str): Path to the image file.
        x_dataset (list): List to store image data.
        y_dataset (list): List to store image labels.
        label (int): Label for the image (0 for dog, 1 for cat).
    """
    try:
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        x_dataset.append(img_arr)
        y_dataset.append(label)
    except Exception as e:
        print(f"{img_path} was not added. Error: {e}")

# --------------------------------------------------------------------------------------
# End of Script
# --------------------------------------------------------------------------------------

# SUMMARY:
# This script loads and preprocesses image data from the specified directory.
#
# USERS:
# Users can call the `load_data` function to load and preprocess image data for training and testing.
