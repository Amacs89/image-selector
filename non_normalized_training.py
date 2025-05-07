import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Képek fix méretre állítás nélkül (nem normalizálunk)
def load_images(directory, target_size):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                filepath = os.path.join(label_path, filename)
                img = tf.keras.preprocessing.image.load_img(filepath, target_size=target_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                # Nem normalizáljuk a pixelértékeket
                images.append(img_array)
                labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Egyszerű modell definíció
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Bináris osztályozás
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Adatok betöltése
train_dir = "path/to/train"
val_dir = "path/to/validation"
test_dir = "path/to/test"

target_size = (150, 150)  # Fix méret
x_train, y_train = load_images(train_dir, target_size)
x_val, y_val = load_images(val_dir, target_size)

# Modell létrehozása és tanítása
model = build_model((150, 150, 3))
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

# Predikció
x_test, y_test = load_images(test_dir, target_size)
predictions = model.predict(x_test)