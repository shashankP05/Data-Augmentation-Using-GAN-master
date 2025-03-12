import os
import cv2
import numpy as np

# Define paths
input_dir = r'C:\Users\prath\OneDrive\Desktop\5\Test'
output_dir = 'generated_images'
num_images = int(input("Enter number of new images to generate: "))

# Function to read and prepare input images
def read_images_from_dir(directory):
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize images to a common size (e.g., 256x256)
                    img = cv2.resize(img, (256, 256))  # Adjust size as needed
                    images.append(img)
    return images

# Function to generate images using image blending
def generate_blended_images(input_images, num_images):
    generated_images = []
    num_input_images = len(input_images)

    for _ in range(num_images):
        # Randomly select two input images
        idx1, idx2 = np.random.choice(num_input_images, size=2, replace=False)
        img1 = input_images[idx1]
        img2 = input_images[idx2]

        # Blend or interpolate between img1 and img2
        alpha = np.random.uniform(0.2, 0.8)  # Example: Use a random blending factor
        new_image = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

        generated_images.append(new_image)

    return generated_images

# Function to generate images using rotation
def generate_rotated_images(input_images, num_images):
    generated_images = []
    num_input_images = len(input_images)

    for _ in range(num_images):
        # Randomly select one input image
        idx = np.random.choice(num_input_images)
        img = input_images[idx]

        # Rotate the image by a random angle
        angle = np.random.uniform(-30, 30)  # Example: Rotate by a random angle between -30 and 30 degrees
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
        new_image = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        generated_images.append(new_image)

    return generated_images

# Function to generate images using flipping
def generate_flipped_images(input_images, num_images):
    generated_images = []
    num_input_images = len(input_images)

    for _ in range(num_images):
        # Randomly select one input image
        idx = np.random.choice(num_input_images)
        img = input_images[idx]

        # Flip the image randomly (horizontally, vertically, or both)
        flip_code = np.random.choice([-1, 0, 1])  # -1: both axes, 0: vertical, 1: horizontal
        new_image = cv2.flip(img, flip_code)

        generated_images.append(new_image)

    return generated_images

# Function to generate images using noise addition
def generate_noisy_images(input_images, num_images):
    generated_images = []
    num_input_images = len(input_images)

    for _ in range(num_images):
        # Randomly select one input image
        idx = np.random.choice(num_input_images)
        img = input_images[idx]

        # Add Gaussian noise to the image
        noise = np.random.normal(0, 25, img.shape)  # Example: Gaussian noise with mean 0 and standard deviation 25
        noisy_image = img + noise

        # Clip the values to be in the proper range
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        generated_images.append(noisy_image)

    return generated_images

# Function to generate images using color adjustments
def generate_color_adjusted_images(input_images, num_images):
    generated_images = []
    num_input_images = len(input_images)

    for _ in range(num_images):
        # Randomly select one input image
        idx = np.random.choice(num_input_images)
        img = input_images[idx]

        # Adjust brightness and contrast randomly
        brightness = np.random.uniform(0.7, 1.3)  # Example: Scale brightness by a random factor between 0.7 and 1.3
        contrast = np.random.uniform(0.7, 1.3)  # Example: Scale contrast by a random factor between 0.7 and 1.3
        new_image = cv2.convertScaleAbs(img, alpha=contrast, beta=(brightness - 1) * 255)

        generated_images.append(new_image)

    return generated_images

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read input images
original_images = read_images_from_dir(input_dir)

# Generate new images using various techniques
techniques = {
    'blended': generate_blended_images,
    'rotated': generate_rotated_images,
    'flipped': generate_flipped_images,
    'noisy': generate_noisy_images,
    'color_adjusted': generate_color_adjusted_images
}

for technique, generate_function in techniques.items():
    technique_dir = os.path.join(output_dir, technique)
    if not os.path.exists(technique_dir):
        os.makedirs(technique_dir)

    generated_images = generate_function(original_images, num_images)

    # Save generated images to output directory for the technique
    for i, img in enumerate(generated_images):
        output_path = os.path.join(technique_dir, f"{technique}_image_{i}.jpg")
        cv2.imwrite(output_path, img)

print(f"{num_images} images generated for each technique and saved in {output_dir}.")
