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
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torchvision.models import resnet50
import random

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU")
    return device


class AdaptiveAugmentation:
    def __init__(self, target_p=0.6, speed=1e-7):
        self.p = 0
        self.target_p = target_p
        self.speed = speed

    def update(self, real_sign):
        self.p += self.speed * (real_sign - self.target_p)
        self.p = min(1.0, max(0.0, self.p))
        return self.p


class DifferentiableAugment:
    @staticmethod
    def rand_brightness(x):
        return x * torch.rand(x.size(0), 1, 1, 1, device=x.device).uniform_(0.5, 1.5)

    @staticmethod
    def rand_saturation(x):
        x_mean = x.mean(dim=1, keepdim=True)
        return (x - x_mean) * torch.rand(x.size(0), 1, 1, 1, device=x.device).uniform_(0, 2) + x_mean

    @staticmethod
    def rand_contrast(x):
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        return (x - x_mean) * torch.rand(x.size(0), 1, 1, 1, device=x.device).uniform_(0.5, 1.5) + x_mean

    def __call__(self, x, p=0.5):
        if random.random() < p:
            functions = [self.rand_brightness, self.rand_contrast, self.rand_saturation]
            random.shuffle(functions)
            for func in functions[:random.randint(1, 3)]:
                x = func(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=2)
        value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x


class Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),

            SelfAttention(ngf * 4),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),

            SelfAttention(ndf * 4),

            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.main(input)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(pretrained=True)
        # Use only the first few layers of ResNet
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:5])
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) * 0.5
        # Normalize with ImageNet stats
        return (x - self.mean) / self.std

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        return self.criterion(x_features, y_features)


def train_gan(dataloader, device, num_epochs=500, save_interval=50):
    set_seed(42)

    nz = 128
    lr_g = 0.0002
    lr_d = 0.0004
    betas = (0.5, 0.999)

    netG = Generator(nz).to(device)
    netD = Discriminator().to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=betas)
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=betas)

    schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=num_epochs)
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=num_epochs)

    criterion = nn.BCELoss()
    perceptual_loss = PerceptualLoss().to(device)
    augment = DifferentiableAugment()
    ada = AdaptiveAugmentation()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Train Discriminator
            netD.zero_grad()
            label_real = torch.ones(batch_size, device=device)
            label_fake = torch.zeros(batch_size, device=device)

            label_real = label_real - 0.1 * torch.rand(label_real.shape, device=device)
            label_fake = label_fake + 0.1 * torch.rand(label_fake.shape, device=device)

            p = ada.update(torch.sign(netD(real_images)).mean().item())
            real_images_aug = augment(real_images, p=p)

            output_real = netD(real_images_aug).view(-1)
            errD_real = criterion(output_real, label_real)

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            fake_images_aug = augment(fake_images.detach(), p=p)

            output_fake = netD(fake_images_aug).view(-1)
            errD_fake = criterion(output_fake, label_fake)

            errD = errD_real + errD_fake
            errD.backward()

            # Gradient penalty
            alpha = torch.rand(batch_size, 1, 1, 1, device=device)
            interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
            output_interpolated = netD(interpolated)
            gradients = torch.autograd.grad(
                outputs=output_interpolated,
                inputs=interpolated,
                grad_outputs=torch.ones_like(output_interpolated),
                create_graph=True,
                retain_graph=True
            )[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            errD += 10 * gradient_penalty
            optimizerD.step()

            # Train Generator
            if i % 1 == 0:
                netG.zero_grad()
                output_fake = netD(fake_images).view(-1)
                errG = criterion(output_fake, label_real)

                try:
                    # Add perceptual loss with error handling
                    p_loss = perceptual_loss(fake_images, real_images)
                    errG += 0.1 * p_loss
                except Exception as e:
                    print(f"Warning: Perceptual loss calculation failed: {e}")
                    p_loss = torch.tensor(0.0, device=device)

                errG.backward()
                optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'P_Loss: {p_loss.item():.4f} Ada_p: {p:.4f}')

        schedulerD.step()
        schedulerG.step()

        if epoch % save_interval == 0:
            with torch.no_grad():
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                                  f'generated_images/fake_epoch_{epoch}.png',
                                  normalize=True,
                                  nrow=8)

    return netG, netD
def main():
    device = setup_gpu()

    # Enhanced image transformations
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    os.makedirs('generated_images', exist_ok=True)

    # Dataset and DataLoader setup with num_workers optimized for your system
    dataset = ImageFolderDataset(root=r"C:\Users\prath\OneDrive\Desktop\5\Test", transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    netG, netD = train_gan(dataloader, device)

    # Save models
    torch.save({
        'generator_state_dict': netG.state_dict(),
        'discriminator_state_dict': netD.state_dict(),
    }, 'gan_checkpoint.pth')

    # Generate new images
    num_images = int(input("Enter number of new images to generate: "))
    os.makedirs('final_generated_images', exist_ok=True)

    netG.eval()
    with torch.no_grad():
        for i in range(num_images):
            noise = torch.randn(1, 128, 1, 1, device=device)
            fake_image = netG(noise)
            vutils.save_image(
                fake_image.detach(),
                f'final_generated_images/generated_image_{i + 1}.png',
                normalize=True
            )
    print(f"Successfully generated {num_images} images in 'final_generated_images' directory")


# Custom Dataset class that was referenced but not defined in the original code
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [
            os.path.join(root, f) for f in os.listdir(root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


if __name__ == "__main__":
    main()