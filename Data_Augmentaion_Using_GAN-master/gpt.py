import os
import sys
import logging
import traceback
from PIL import Image
import numpy as np
from skimage import img_as_ubyte

# Constants
output_dir = 'output8'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def get_images_from_folder(folder: str) -> list:
    '''Get a list of image file paths from a specified folder.'''
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            images.append(os.path.join(folder, filename))
    return images


def generate_similar_images(image_files: list, num_images: int, output_dir: str) -> list:
    '''Generate a specified number of similar images based on the existing images.'''
    if len(image_files) < 2:
        raise ValueError("Not enough images to morph between.")

    similar_images = []
    for i in range(num_images):
        image1 = image_files[i % len(image_files)]
        image2 = image_files[(i + 1) % len(image_files)]
        morphed_images, _ = morph_images(image1, image2, steps=num_images, output_dir=output_dir,
                                         start_index=i * num_images)
        similar_images.extend(morphed_images)

    return similar_images


def morph_images(image1: str, image2: str, steps: int = 10, output_dir: str = ".", start_index: int = 0) -> list:
    '''Morph images in number of steps.'''
    try:
        with Image.open(image1) as img1, Image.open(image2) as img2:
            img1_array = np.array(img1, dtype=np.float32) / 255.0
            img2_array = np.array(img2, dtype=np.float32) / 255.0

        morphed_images = []
        for cnt in range(steps + 1):
            alpha = cnt / steps
            morphed_image = (1 - alpha) * img1_array + alpha * img2_array

            # Save the morphed image to the disk
            val = start_index + cnt
            filename = f"{output_dir}/morphed_image_{val:04d}.png"
            img = Image.fromarray(img_as_ubyte(morphed_image))
            img.save(filename)
            morphed_images.append(filename)

        return morphed_images, val + 1

    except Exception as e:
        logging.error(f"An error occurred while morphing images: {e}\n{traceback.format_exc()}")
        return []


def main():
    folder_path =r'C:\Users\prath\OneDrive\Desktop\5\Test'
    if not os.path.isdir(folder_path):
        print(f"Invalid folder path: {folder_path}")
        return

    num_images = int(input("Enter the number of new images to generate: "))
    if num_images < 1:
        print("Number of images must be at least 1.")
        return

    image_files = get_images_from_folder(folder_path)
    if not image_files:
        print("No images found in the specified folder.")
        return

    similar_images = generate_similar_images(image_files, num_images, output_dir)
    print(f"Generated {len(similar_images)} similar images.")


if __name__ == "__main__":
    # Setup logging file
    name, _ = os.path.splitext(sys.argv[0])
    name += '_.log'
    # Set up logging to file
    logging.basicConfig(
        level=logging.INFO,
        filename=(name), filemode='w',
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%d-%H:%M:%S'
    )
    main()
