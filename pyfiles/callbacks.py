# File: callbacks.py
# Purpose: Define custom Keras callback to update Streamlit UI during training

# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Developer details:
#   Name         : Lay Sheth and Rishav Raj
#   Role         : Software Engineers
#   Version      : V 1.0
#   Unit test    : Pass
#   Integration test: Pass
#   Description  : This script defines a custom Keras callback to update Streamlit UI during training.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import streamlit as st
from tensorflow.keras.callbacks import Callback
import plotly.graph_objects as go

class StreamlitCallback(Callback):
    """
    Custom Keras callback to update Streamlit UI during training.

    Args:
        epochs (int): Total number of epochs for training.
        batch_size (int): Batch size for training.
    """
    def __init__(self, epochs, batch_size):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.epoch_progress = st.empty()
        self.loss_chart = st.empty()
        self.accuracy_chart = st.empty()
        self.val_loss_chart = st.empty()
        self.val_accuracy_chart = st.empty()
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        """
        Update the Streamlit UI at the end of each epoch.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary of logs from the current epoch.
        """
        self.progress_bar.progress((epoch + 1) / self.epochs)
        self.status_text.text(f"Epoch {epoch + 1}/{self.epochs}")
        self.epoch_progress.text(f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, "
                                 f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")

        # Update history
        self.history['loss'].append(logs['loss'])
        self.history['accuracy'].append(logs['accuracy'])
        self.history['val_loss'].append(logs['val_loss'])
        self.history['val_accuracy'].append(logs['val_accuracy'])

        # Update charts
        self.update_charts()

    def update_charts(self):
        """
        Update the charts with the latest training metrics.
        """
        epochs = list(range(1, len(self.history['loss']) + 1))

        # Loss chart
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=epochs, y=self.history['loss'], mode='lines+markers', name='Loss'))
        loss_fig.add_trace(go.Scatter(x=epochs, y=self.history['val_loss'], mode='lines+markers', name='Val Loss'))
        loss_fig.update_layout(title='Loss per Epoch', xaxis_title='Epoch', yaxis_title='Loss')
        self.loss_chart.plotly_chart(loss_fig)

        # Accuracy chart
        accuracy_fig = go.Figure()
        accuracy_fig.add_trace(go.Scatter(x=epochs, y=self.history['accuracy'], mode='lines+markers', name='Accuracy'))
        accuracy_fig.add_trace(go.Scatter(x=epochs, y=self.history['val_accuracy'], mode='lines+markers', name='Val Accuracy'))
        accuracy_fig.update_layout(title='Accuracy per Epoch', xaxis_title='Epoch', yaxis_title='Accuracy')
        self.accuracy_chart.plotly_chart(accuracy_fig)

# --------------------------------------------------------------------------------------
# End of Script
# --------------------------------------------------------------------------------------

# SUMMARY:
# This script defines a custom Keras callback to update Streamlit UI during training.
#
# USERS:
# Users can use the `StreamlitCallback` class to update the Streamlit UI with training progress.
