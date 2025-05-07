import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_squared_error
import shutil

# Autoencoder model definition
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Flatten()(input_img)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    encoded = Dense(32, activation='relu')(x)

    x = Dense(64, activation='relu')(encoded)
    x = Dense(128, activation='relu')(x)
    x = Dense(np.prod(input_shape), activation='sigmoid')(x)
    decoded = Reshape(input_shape)(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Prepare and normalize data
def prepare_data(data_dir, img_height, img_width):
    datagen = ImageDataGenerator(rescale=1./255)
    data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode=None,
        shuffle=True)
    return data_generator

# Train Autoencoder
def train_autoencoder(autoencoder, data_generator, epochs=10):
    autoencoder.fit(data_generator, epochs=epochs)
    return autoencoder

# Detect anomalies
def detect_anomalies(autoencoder, target_generator, output_dir, threshold=0.01):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (img, path) in enumerate(zip(target_generator, target_generator.filenames)):
        # Reconstruct the image
        reconstructed = autoencoder.predict(img)
        mse = mean_squared_error(img.flatten(), reconstructed.flatten())

        # If the MSE is above the threshold, classify as anomalous
        if mse < threshold:
            shutil.copy(path, output_dir)

if __name__ == "__main__":
    # Paths
    positive_data_dir = "path/to/positive_samples"
    target_data_dir = "path/to/target_images"
    output_dir = "path/to/output"

    # Parameters
    img_height, img_width = 150, 150

    # Prepare data
    positive_data = prepare_data(positive_data_dir, img_height, img_width)

    # Build and train the model
    autoencoder = build_autoencoder((img_height, img_width, 3))
    autoencoder = train_autoencoder(autoencoder, positive_data, epochs=10)

    # Detect anomalies
    target_data = prepare_data(target_data_dir, img_height, img_width)
    detect_anomalies(autoencoder, target_data, output_dir)