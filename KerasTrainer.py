import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# Constants
IMG_SIZE = 256
CLASSES = {0: "Dog", 1: "Cat"}

def load_data(path):
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
    try:
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        x_dataset.append(img_arr)
        y_dataset.append(label)
    except Exception as e:
        print(f"{img_path} was not added. Error: {e}")

def create_model(learning_rate):
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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to open.")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype('float32') / 255.0
    return np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 1))

def display_prediction(image_path, predicted_class, prediction):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to open.")
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_class}\nProbability: {prediction[0][0]:.4f}")
    plt.axis('off')
    st.pyplot(plt)

class StreamlitCallback(Callback):
    def __init__(self, epochs, batch_size):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.epoch_progress = st.empty()
        self.loss_chart = st.line_chart()
        self.accuracy_chart = st.line_chart()
        self.val_loss_chart = st.line_chart()
        self.val_accuracy_chart = st.line_chart()
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
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
        self.loss_chart.add_rows([logs['loss']])
        self.accuracy_chart.add_rows([logs['accuracy']])
        self.val_loss_chart.add_rows([logs['val_loss']])
        self.val_accuracy_chart.add_rows([logs['val_accuracy']])

def main():
    st.title("Cats and Dogs Classifier")

    # Sidebar for user inputs
    st.sidebar.header("Configuration")
    train_path = st.sidebar.text_input("Training Data Path", "dataset/training_set/training_set")
    test_path = st.sidebar.text_input("Testing Data Path", "dataset/test_set/test_set")
    learning_rate = st.sidebar.radio("Learning Rate", [0.0001, 0.001, 0.01])
    epochs = st.sidebar.slider("Epochs", 1, 50, 5)
    batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)

    if st.sidebar.button("Train Model"):
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
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[streamlit_callback])

        st.write("Evaluating model...")
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
        st.write(f"Test accuracy: {test_accuracy}")

        st.write("Plotting training history...")
        plot_training_history(history)

        st.write("Saving model...")
        model.save("dogcatclassifier.h5")

    st.sidebar.header("Prediction")
    sample_image_path = st.sidebar.text_input("Sample Image Path", "images/dog1.jpg")
    if st.sidebar.button("Predict"):
        model = load_model("dogcatclassifier.h5")
        prediction = model.predict(preprocess_image(sample_image_path))
        predicted_class = CLASSES[round(prediction[0][0])]
        display_prediction(sample_image_path, predicted_class, prediction)

if __name__ == "__main__":
    main()
