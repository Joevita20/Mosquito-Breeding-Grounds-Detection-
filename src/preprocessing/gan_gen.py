"""
gan_gen.py
----------
DCGAN implementation to generate synthetic images for under-represented 
classes (e.g., "pools") in the MBG dataset, as described in the paper.

Usage:
    python3 src/preprocessing/gan_gen.py \
        --real_dir data/augmented/pools \
        --output_dir data/gan_generated/pools \
        --num_images 200 \
        --epochs 200

Note:
    Requires: torch, torchvision
    Best run on Google Colab with GPU.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

# Hyperparameters
LATENT_DIM = 100
IMAGE_SIZE = 64  # GAN generates 64x64; upscale to 640x640 for dataset
CHANNELS = 3
FEATURES_G = 64
FEATURES_D = 64


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            self._block(latent_dim, FEATURES_G * 16, 4, 1, 0),
            self._block(FEATURES_G * 16, FEATURES_G * 8, 4, 2, 1),
            self._block(FEATURES_G * 8, FEATURES_G * 4, 4, 2, 1),
            self._block(FEATURES_G * 4, FEATURES_G * 2, 4, 2, 1),
            nn.ConvTranspose2d(FEATURES_G * 2, CHANNELS, 4, 2, 1),
            nn.Tanh(),
        )

    def _block(self, in_ch, out_ch, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, FEATURES_D, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(FEATURES_D, FEATURES_D * 2, 4, 2, 1),
            self._block(FEATURES_D * 2, FEATURES_D * 4, 4, 2, 1),
            self._block(FEATURES_D * 4, FEATURES_D * 8, 4, 2, 1),
            nn.Conv2d(FEATURES_D * 8, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def _block(self, in_ch, out_ch, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)


def train_gan(real_dir: str, output_dir: str, num_images: int, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = datasets.ImageFolder(root=os.path.dirname(real_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(device)
            b = real_imgs.size(0)

            # --- Train Discriminator ---
            D.zero_grad()
            real_labels = torch.ones(b, 1, 1, 1, device=device)
            fake_labels = torch.zeros(b, 1, 1, 1, device=device)

            loss_D_real = criterion(D(real_imgs), real_labels)
            z = torch.randn(b, LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(z)
            loss_D_fake = criterion(D(fake_imgs.detach()), fake_labels)
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            opt_D.step()

            # --- Train Generator ---
            G.zero_grad()
            loss_G = criterion(D(fake_imgs), real_labels)
            loss_G.backward()
            opt_G.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

    # Generate final synthetic images
    os.makedirs(output_dir, exist_ok=True)
    G.eval()
    with torch.no_grad():
        for i in range(0, num_images, 16):
            n = min(16, num_images - i)
            z = torch.randn(n, LATENT_DIM, 1, 1, device=device)
            imgs = G(z) * 0.5 + 0.5
            for j, img in enumerate(imgs):
                utils.save_image(img, os.path.join(output_dir, f"gan_pool_{i+j:04d}.jpg"))

    print(f"[INFO] Generated {num_images} synthetic images -> {output_dir}")
    torch.save(G.state_dict(), os.path.join(output_dir, "generator.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCGAN for MBG minority classes.")
    parser.add_argument("--real_dir", type=str, default="data/augmented/pools",
                        help="Directory with real pool images.")
    parser.add_argument("--output_dir", type=str, default="data/gan_generated/pools")
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    train_gan(args.real_dir, args.output_dir, args.num_images, args.epochs)
