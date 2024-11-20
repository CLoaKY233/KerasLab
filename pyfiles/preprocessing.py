# File: preprocessing.py
# Purpose: Preprocess images for prediction

# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Developer details:
#   Name         : Lay Sheth and Rishav Raj
#   Role         : Software Engineers
#   Version      : V 1.0
#   Unit test    : Pass
#   Integration test: Pass
#   Description  : This script preprocesses images for prediction.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import cv2
import numpy as np
from pyfiles.constants import IMG_SIZE

def preprocess_image(image_path):
    """
    Preprocess a single image for prediction.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to open.")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype('float32') / 255.0
    return np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 1))

# --------------------------------------------------------------------------------------
# End of Script
# --------------------------------------------------------------------------------------

# SUMMARY:
# This script preprocesses images for prediction.
#
# USERS:
# Users can call the `preprocess_image` function to preprocess images for prediction.
