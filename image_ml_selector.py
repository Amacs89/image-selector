import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

def prepare_data(target_dir, img_height, img_width):
    """
    Prepares image data for prediction
    """
    datagen = ImageDataGenerator(rescale=1./255)
    data_generator = datagen.flow_from_directory(
        target_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    return data_generator

def predict_images(model_path, target_dir, output_dir, img_height=150, img_width=150, threshold=0.5):
    """
    Predicts images using a pre-trained model and copies matching images to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the trained model
    model = load_model(model_path)

    # Prepare data
    data_generator = prepare_data(target_dir, img_height, img_width)

    # Predict and copy images
    for i, (img_path, img_array) in enumerate(zip(data_generator.filenames, data_generator)):
        prediction = model.predict(img_array)
        if prediction[0][0] > threshold:
            source_path = os.path.join(target_dir, img_path)
            dest_path = os.path.join(output_dir, os.path.basename(img_path))
            shutil.copy2(source_path, dest_path)
            print(f"Copied {img_path} to {output_dir}")

if __name__ == "__main__":
    # Paths
    model_path = "path/to/your/model.h5"  # Path to your pre-trained model
    target_dir = "path/to/target_images"  # Path to the directory containing images to evaluate
    output_dir = "path/to/output_images"  # Path to store matched images

    # Model parameters
    img_height, img_width = 150, 150
    threshold = 0.5  # Confidence threshold for classification

    predict_images(model_path, target_dir, output_dir, img_height, img_width, threshold)