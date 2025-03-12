from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.utils as vutils
import os
import zipfile
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from io import BytesIO

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
netG.load_state_dict(torch.load('generator_final.pth'))
netG.eval()

@app.route('/generate-images', methods=['POST'])
def generate_images():
    data = request.json
    email = data.get('email')
    num_images = int(data.get('num_images'))

    # Create a directory to save generated images
    output_dir = 'images'
    os.makedirs(output_dir, exist_ok=True)

    # Generate new images
    for i in range(num_images):
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake_image = netG(noise)
        vutils.save_image(fake_image.detach(), os.path.join(output_dir, f'new_generated_image_{i}.png'), normalize=True)

    # Zip the generated images
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            zip_file.write(file_path, file_name)

    zip_buffer.seek(0)  # Reset buffer position to the beginning

    # Send email with the zipped folder
    send_email(email, zip_buffer)

    return jsonify({'status': 'success'})

def send_email(recipient_email, zip_buffer):
    from_email = 'prathviiastria@gmail.com'  # Replace with your email
    from_password = 'mkdk xeey xtsr aosr'  # Replace with your app-specific password
    subject = 'Generated Images'
    body = 'Please find attached the generated images.'

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    part = MIMEBase('application', 'zip')
    part.set_payload(zip_buffer.getvalue())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="images.zip"')
    msg.attach(part)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:  # Replace with your SMTP server details
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, recipient_email, msg.as_string())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
