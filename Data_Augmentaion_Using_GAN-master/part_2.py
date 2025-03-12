import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from torch.nn import init
import warnings

warnings.filterwarnings('ignore')

# Enhanced GPU setup
def setup_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU")
    return device

# Custom dataset class with error handling
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.images:
            raise RuntimeError(f"No images found in {root}")
        print(f"Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {self.images[idx]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

# Improved Generator with Xavier initialization
class Generator(nn.Module):
    def __init__(self, nz, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            init.xavier_normal_(m.weight.data)

    def forward(self, input):
        return self.main(input)

# Improved Discriminator with spectral normalization
class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_normal_(m.weight.data)

    def forward(self, input):
        return self.main(input)

def train_gan(dataloader, device, num_epochs=500, save_interval=50):
    nz = 128  # Latent vector size
    lr = 0.0001  # Fine-tuned learning rate
    beta1 = 0.5  # Adam optimizer parameter

    netG = Generator(nz).to(device)
    netD = Discriminator().to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            netD.zero_grad()
            real_images = data.to(device)
            batch_size = real_images.size(0)
            label_real = torch.full((batch_size,), 1., device=device)
            label_fake = torch.full((batch_size,), 0., device=device)

            output_real = netD(real_images).view(-1)
            errD_real = criterion(output_real, label_real)

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output_fake, label_fake)
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            netG.zero_grad()
            output_fake = netD(fake_images).view(-1)
            errG = criterion(output_fake, label_real)
            errG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

        if epoch % save_interval == 0:
            with torch.no_grad():
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(), f'generated_images/fake_epoch_{epoch}.png', normalize=True)

    return netG, netD

def main():
    device = setup_gpu()

    batch_size = 128
    image_size = 64
    num_epochs = 500

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    os.makedirs('generated_images', exist_ok=True)

    dataset = ImageFolderDataset(root=r"C:\Users\prath\OneDrive\Desktop\5\Test", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    netG, netD = train_gan(dataloader, device, num_epochs)
    torch.save(netG.state_dict(), 'generator_final.pth')
    torch.save(netD.state_dict(), 'discriminator_final.pth')

    num_new_images = int(input("Enter number of new images to generate: "))
    os.makedirs('new_generated_images', exist_ok=True)

    netG.eval()
    with torch.no_grad():
        for i in range(num_new_images):
            noise = torch.randn(1, 128, 1, 1, device=device)
            fake_image = netG(noise)
            vutils.save_image(fake_image.detach(), f'new_generated_images/generated_image_{i}.png', normalize=True)

    print(f"Successfully generated {num_new_images} new images.")

if __name__ == "__main__":
    main()
