# File: visualization.py
# Purpose: Plot training history and display prediction results

# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Developer details:
#   Name         : Lay Sheth and Rishav Raj
#   Role         : Software Engineers
#   Version      : V 1.0
#   Unit test    : Pass
#   Integration test: Pass
#   Description  : This script plots the training history and displays prediction results.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import matplotlib.pyplot as plt
import streamlit as st
import cv2
import numpy as np
from pyfiles.constants import IMG_SIZE, CLASSES

def plot_training_history(history):
    """
    Plot the training history of the model.

    Args:
        history (keras.callbacks.History): History object returned by the fit method.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    st.pyplot(fig)

def display_prediction(image_path, predicted_class, prediction):
    """
    Display the prediction result for a single image.

    Args:
        image_path (str): Path to the image file.
        predicted_class (str): Predicted class label.
        prediction (numpy.ndarray): Prediction probability.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to open.")
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_class}\nProbability: {prediction[0][0]:.4f}")
    plt.axis('off')
    st.pyplot(plt)

# --------------------------------------------------------------------------------------
# End of Script
# --------------------------------------------------------------------------------------

# SUMMARY:
# This script plots the training history and displays prediction results.
#
# USERS:
# Users can call the `plot_training_history` function to visualize the training history and
# the `display_prediction` function to display prediction results for a single image.
