import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import plotly.graph_objects as go

# Constants
IMG_SIZE = 256
CLASSES = {0: "Dog", 1: "Cat"}

def load_data(path):
    """
    Load and preprocess image data from the specified directory.

    This function reads images from the given directory, processes them, and returns
    the images and their corresponding labels as numpy arrays. The images are resized
    to a fixed size and converted to grayscale.

    Args:
        path (str): Path to the dataset directory. The directory should contain two
                    subdirectories named 'cats' and 'dogs', each containing the respective images.

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

    This function reads an image from the given path, resizes it to a fixed size,
    converts it to grayscale, and appends it to the dataset along with its label.

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

def create_model(learning_rate):
    """
    Create and compile a Convolutional Neural Network model.

    This function defines the architecture of the CNN model, compiles it with the
    specified learning rate, and returns the compiled model.

    Args:
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        keras.models.Sequential: Compiled CNN model.
    """
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1), padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def plot_training_history(history):
    """
    Plot the training history of the model.

    This function creates two subplots: one for the accuracy and one for the loss
    during training and validation. It then displays these plots using Streamlit.

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

def preprocess_image(image_path):
    """
    Preprocess a single image for prediction.

    This function reads an image from the given path, resizes it to a fixed size,
    converts it to grayscale, normalizes the pixel values, and reshapes it to the
    required input shape for the model.

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

def display_prediction(image_path, predicted_class, prediction):
    """
    Display the prediction result for a single image.

    This function reads an image from the given path, displays it using matplotlib,
    and overlays the predicted class and probability on the image.

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

class StreamlitCallback(Callback):
    """
    Custom Keras callback to update Streamlit UI during training.

    This class defines a custom callback that updates the Streamlit UI with the
    training progress, including the current epoch, loss, accuracy, and validation
    metrics. It also updates real-time charts for loss and accuracy.

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

        This method updates the progress bar, status text, and epoch progress text
        with the current training metrics. It also updates the history of loss and
        accuracy for both training and validation, and refreshes the charts.

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

        This method creates and updates Plotly charts for loss and accuracy using
        the history of training and validation metrics.
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
