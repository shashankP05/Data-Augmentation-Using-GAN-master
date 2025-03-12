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
from torchvision.models import resnet50, efficientnet_b0, inception_v3
import random
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from scipy import linalg

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
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Transform input
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        value = self.value(x).view(batch_size, -1, width * height)

        # Calculate attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        return self.gamma * out + x


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):  # Changed nc to 3 for RGB images
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # state size. (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # state size. (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # Output: 1 x 1 x 1
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        # Add batch dimension if it's missing
        if input.dim() == 3:
            input = input.unsqueeze(0)
        return self.main(input).view(-1, 1).squeeze(1)

class EnhancedGenerator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=3):
        super().__init__()

        self.noise_projection = nn.Linear(nz, nz * 4 * 4)

        self.main = nn.ModuleList([
            # Path 1 - Higher capacity path
            nn.Sequential(
                self._make_gen_block(nz, ngf * 16),
                self._make_gen_block(ngf * 16, ngf * 8),
                self._make_gen_block(ngf * 8, ngf * 4),
                SelfAttention(ngf * 4),
                self._make_gen_block(ngf * 4, nc, final_layer=True)
            ),

            # Path 2 - Lower capacity path
            nn.Sequential(
                self._make_gen_block(nz, ngf * 8),
                self._make_gen_block(ngf * 8, ngf * 4),
                SelfAttention(ngf * 4),
                self._make_gen_block(ngf * 4, ngf * 2),
                self._make_gen_block(ngf * 2, nc, final_layer=True)
            )
        ])

        self.weight_module = nn.Sequential(
            nn.Linear(nz, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        self.apply(self._init_weights)

    def _make_gen_block(self, in_channels, out_channels,ngf=64, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.LeakyReLU(0.2, True),
                nn.Dropout2d(0.2)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )

    def _init_weights(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, z):
        batch_size = z.size(0)

        # Project and reshape noise
        x = self.noise_projection(z.view(batch_size, -1))
        x = x.view(batch_size, -1, 4, 4)

        # Calculate path weights
        weights = self.weight_module(z.view(batch_size, -1))  # Shape: [batch_size, 2]

        # Process through parallel paths
        outputs = []
        for path in self.main:
            path_input = x[:, :self.main[0][0][0].in_channels]
            outputs.append(path(path_input))

        # Reshape weights for broadcasting
        weights = weights.view(batch_size, 2, 1, 1, 1)

        # Combine outputs using weights
        weighted_outputs = [outputs[i] * weights[:, i] for i in range(2)]
        final_output = sum(weighted_outputs)

        return final_output
class EnhancedPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.efficientnet = efficientnet_b0(pretrained=True)

        self.resnet_features = nn.ModuleList([
            nn.Sequential(*list(self.resnet.children())[:5]),
            nn.Sequential(*list(self.resnet.children())[:7])
        ])

        self.efficient_features = nn.ModuleList([
            nn.Sequential(*list(self.efficientnet.features)[:4]),
            nn.Sequential(*list(self.efficientnet.features)[:6])
        ])

        for model in [*self.resnet_features, *self.efficient_features]:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        self.criterion = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        x = (x + 1) * 0.5
        return (x - self.mean) / self.std

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)

        loss = 0
        for res_feat, eff_feat in zip(self.resnet_features, self.efficient_features):
            res_x = res_feat(x)
            res_y = res_feat(y)
            eff_x = eff_feat(x)
            eff_y = eff_feat(y)

            loss += self.criterion(res_x, res_y) + self.criterion(eff_x, eff_y)

        return loss / len(self.resnet_features)


class InceptionStatistics:
    def __init__(self, device):
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

    @torch.no_grad()
    def get_features(self, images):
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        features = self.model(images).squeeze(-1).squeeze(-1)
        return features


def calculate_fid(real_images, fake_images, device='cuda'):
    inception = InceptionStatistics(device)

    def get_statistics(images):
        features = []
        for img_batch in tqdm(images, desc="Calculating statistics"):
            features.append(inception.get_features(img_batch))
        features = torch.cat(features, dim=0).cpu().numpy()

        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    mu1, sigma1 = get_statistics(real_images)
    mu2, sigma2 = get_statistics(fake_images)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def train_enhanced_gan(dataloader, device, num_epochs=500):
    # Hyperparameters
    lr = 0.0002
    beta1 = 0.5
    nz = 128  # Size of z latent vector

    # Initialize networks
    netG = EnhancedGenerator(nz=nz).to(device)
    netD = Discriminator().to(device)

    # Setup loss and optimizers
    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to store losses
    G_losses = []
    D_losses = []
    fid_scores = []

    try:
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader):
                # Get batch size
                real_images = data[0].to(device)
                batch_size = real_images.size(0)

                # Create labels
                real_label = torch.ones(batch_size, device=device)
                fake_label = torch.zeros(batch_size, device=device)

                ############################
                # (1) Update D network
                ###########################
                netD.zero_grad()

                # Train with real
                print("Real images shape:", real_images.shape)
                output = netD(real_images)
                errD_real = criterion(output, real_label)
                errD_real.backward()
                D_x = output.mean().item()

                # Train with fake
                noise = torch.randn(batch_size, nz, device=device)
                fake = netG(noise)
                output = netD(fake.detach())
                errD_fake = criterion(output, fake_label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()
                output = netD(fake)
                errG = criterion(output, real_label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                # Save losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if i % 10 == 0:
                    print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                          f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                          f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

    return netG, netD, G_losses, D_losses, fid_scores

def validate_dataset(dataset_path):
    invalid_images = []
    for img_file in os.listdir(dataset_path):
        if img_file.endswith(('.jpg', '.png')):
            try:
                img_path = os.path.join(dataset_path, img_file)
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                invalid_images.append((img_file, str(e)))

    if invalid_images:
        print("Found invalid images:")
        for img, error in invalid_images:
            print(f"{img}: {error}")
    else:
        print("All images are valid")
# Move the ImageDataset class definition to the top level of the file
class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def main():
    # Set device
    device = setup_gpu()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
    ])

    # Training configuration
    batch_size = 64
    num_workers = 0  # Set to 0 first for debugging

    # Create dataset and dataloader
    dataset = ImageDataset(r"C:\Users\prath\OneDrive\Desktop\5\Test", transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # Train the model
    try:
        netG, netD, G_losses, D_losses, fid_scores = train_enhanced_gan(dataloader, device)

        # Save final model
        torch.save({
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses,
            'fid_scores': fid_scores,
        }, 'final_model.pth')

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e


if __name__ == "__main__":
    main()