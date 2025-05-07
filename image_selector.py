import os
import shutil
from PIL import Image
import imagehash

def get_image_hash(image_path):
    """
    Generate a hash for the given image to enable comparison.
    """
    try:
        with Image.open(image_path) as img:
            return imagehash.average_hash(img)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def find_matching_images(sample_images_dir, target_images_dir, output_dir):
    """
    Find images in the target_images_dir that match any image in sample_images_dir
    and copy them to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sample_hashes = {}
    for root, _, files in os.walk(sample_images_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                sample_path = os.path.join(root, file)
                sample_hash = get_image_hash(sample_path)
                if sample_hash:
                    sample_hashes[file] = sample_hash

    for root, _, files in os.walk(target_images_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                target_path = os.path.join(root, file)
                target_hash = get_image_hash(target_path)
                if target_hash and target_hash in sample_hashes.values():
                    output_path = os.path.join(output_dir, file)
                    shutil.copy2(target_path, output_path)
                    print(f"Copied {file} to {output_dir}")

if __name__ == "__main__":
    # Paths
    sample_images_dir = "path/to/sample_images"
    target_images_dir = "path/to/target_images"
    output_dir = "path/to/output"

    find_matching_images(sample_images_dir, target_images_dir, output_dir)