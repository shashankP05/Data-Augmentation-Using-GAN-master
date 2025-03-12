import torch
import torch.nn as nn
import torchvision.utils as vutils
import os

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

# Create a directory to save generated images
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)

# Generate new images
num_new_images = int(input("Enter number of new images to generate: "))
for i in range(num_new_images):
    noise = torch.randn(1, nz, 1, 1, device=device)
    fake_image = netG(noise)
    vutils.save_image(fake_image.detach(), os.path.join(output_dir, f'new_generated_image_{i}.png'), normalize=True)

print(f"Generated {num_new_images} new images and saved them in the '{output_dir}' folder.")
