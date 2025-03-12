from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.utils as vutils
import os
import zipfile
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from io import BytesIO
import shutil
import cv2
import numpy as np

app = Flask(__name__)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Generator
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Hyperparameters
nz = 128  # Updated to match the training configuration

# Initialize the generator
netG = Generator(nz).to(device)

# Load the generator model
netG.load_state_dict(torch.load('generator1.pth'))
netG.eval()

@app.route('/generate-images', methods=['POST'])
def generate_images():
    email = request.form['email']
    num_images = int(request.form['num_images'])

    # Handle ZIP file
    if 'zip_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No ZIP file part'}), 400

    zip_file = request.files['zip_file']
    if zip_file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    # Save the ZIP file
    zip_path = 'uploaded.zip'
    zip_file.save(zip_path)

    # Unzip the file
    input_dir = 'extracted_images'
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(input_dir)

    # Read and process images
    original_images = read_images_from_dir(input_dir)

    # Generate new images using various techniques
    techniques = {
        'blended': generate_blended_images,
        'rotated': generate_rotated_images,
        'flipped': generate_flipped_images,
        'noisy': generate_noisy_images,
        'color_adjusted': generate_color_adjusted_images
    }

    output_dir = 'generated_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for technique, generate_function in techniques.items():
        technique_dir = os.path.join(output_dir, technique)
        if not os.path.exists(technique_dir):
            os.makedirs(technique_dir)

        generated_images = generate_function(original_images, num_images)

        for i, img in enumerate(generated_images):
            output_path = os.path.join(technique_dir, f"{technique}_image_{i}.jpg")
            cv2.imwrite(output_path, img)

    # Zip the generated images
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for foldername, subfolders, filenames in os.walk(output_dir):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, output_dir)
                zip_file.write(file_path, arcname)

    # Send email with the zipped folder
    send_email(email, zip_buffer)

    return jsonify({'status': 'success'})

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

def send_email(recipient_email, zip_buffer):
    from_email = 'prathviiastria@gmail.com'  # Replace with your email
    from_password = 'mkdk xeey xtsr aosr'  # Replace with your app-specific password
    subject = 'Generated Images'
    body = 'Please find attached the generated images.'

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    part = MIMEBase('application', 'zip')
    part.set_payload(zip_buffer.getvalue())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="images.zip"')
    msg.attach(part)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(from_email, from_password)  # Use app password here
        server.sendmail(from_email, recipient_email, msg.as_string())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
