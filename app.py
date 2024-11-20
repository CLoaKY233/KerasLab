# File: app.py
# Purpose: Main function to run the Streamlit app

# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Developer details:
#   Name         : Lay Sheth and Rishav Raj
#   Role         : Software Engineers
#   Version      : V 1.0
#   Unit test    : Pass
#   Integration test: Pass
#   Description  : This script contains the main function to run the Streamlit app.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import streamlit as st
from tensorflow.keras.models import load_model
from pyfiles.data_loader import load_data
from pyfiles.model import create_model
from pyfiles.visualization import plot_training_history, display_prediction
from pyfiles.preprocessing import preprocess_image
from pyfiles.callbacks import StreamlitCallback
from pyfiles.constants import IMG_SIZE, CLASSES

def main():
    """
    Main function to run the Streamlit app.

    This function sets up the Streamlit UI, including the configuration and training
    sections. It allows users to specify data paths, training parameters, and start
    the training process. It also includes a section for making predictions on new images.
    """
    # Set wide mode by default
    st.set_page_config(layout="wide")

    st.title("Cats and Dogs Classifier")

    # Main columns
    a, b = st.columns([1, 1])

    # Training Analytics Column (Right)
    with b:
        analytics_placeholder = st.empty()
        with analytics_placeholder.container(border=True, height=850):
            st.header("Training Analytics")
            st.markdown("---")  # Visual separator
            if 'training_started' not in st.session_state:
                st.write("Click 'Start Training' to begin the training process.")

    # Configuration Column (Left)
    with a:
        with st.container(border=True, height=850):
            st.header("Configuration")
            st.markdown("---")  # Visual separator

            # Data paths section
            st.subheader("Data Paths")
            train_path = st.text_input("Training Data Path", "dataset/training_set/training_set")
            test_path = st.text_input("Testing Data Path", "dataset/test_set/test_set")

            st.markdown("---")  # Visual separator

            # Training parameters section
            st.subheader("Training Parameters")
            col1, col2 = st.columns(2)
            with col1:
                learning_rate = st.selectbox("Learning Rate", [0.0001, 0.001, 0.01])
            with col2:
                epochs = st.slider("Epochs", 1, 50, 5)
                batch_size = st.slider("Batch Size", 16, 128, 32)

            st.markdown("---")  # Visual separator

            # Training button with some padding
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Start Training", use_container_width=True):
                analytics_placeholder.empty()
                with analytics_placeholder.container(border=True, height=850):
                    st.header("Training Analytics")
                    st.markdown("---")
                    with st.spinner("Training started..."):
                        st.write("Loading and preprocessing data...")
                        x_train, y_train = load_data(train_path)
                        x_test, y_test = load_data(test_path)
                        x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
                        x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
                        y_train = y_train.reshape(-1, 1)
                        y_test = y_test.reshape(-1, 1)

                        st.write("Creating and training model...")
                        model = create_model(learning_rate)
                        streamlit_callback = StreamlitCallback(epochs, batch_size)
                        history = model.fit(x_train, y_train,
                                          validation_data=(x_test, y_test),
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          callbacks=[streamlit_callback])

                        st.write("Evaluating model...")
                        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
                        st.write(f"Test accuracy: {test_accuracy}")

                        with st.container(border=True):
                            st.subheader("Training History")
                            plot_training_history(history)

                            st.write("Saving model...")
                            model.save("dogcatclassifier.h5")

                        st.session_state.training_started = True

    # Prediction section in main area instead of sidebar
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    _, pred_col2,_ = st.columns([1,9, 1])
    with pred_col2:
        with st.container(border=True):
            st.header("Model Prediction")
            st.markdown("---")
            sample_image_path = st.text_input("Sample Image Path", "images/cat1.jpg")
            if st.button("Predict", use_container_width=True):
                model = load_model("dogcatclassifier.h5")
                prediction = model.predict(preprocess_image(sample_image_path))
                predicted_class = CLASSES[round(prediction[0][0])]
                display_prediction(sample_image_path, predicted_class, prediction)

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------
# End of Script
# --------------------------------------------------------------------------------------

# SUMMARY:
# This script contains the main function to run the Streamlit app. It sets up the UI,
# allows users to configure training parameters, start the training process, and make predictions.
#
# USERS:
# Users can run this script to start the Streamlit app and interact with the model.
