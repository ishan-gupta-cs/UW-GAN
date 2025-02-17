import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from models.generator import Generator
from models.discriminator import Discriminator
from dataloader import get_dataloader
from torch.optim import Adam
from losses.gan_loss import GANLoss
import os
import yaml

def calculate_psnr(img1, img2):
    img1 = img1.cpu().numpy().astype(np.float32) * 255.0
    img2 = img2.cpu().numpy().astype(np.float32) * 255.0
    return cv2.PSNR(img1, img2)

def calculate_ssim(img1, img2):
    img1 = img1.squeeze().numpy()  # Convert from tensor to NumPy array
    img2 = img2.squeeze().numpy()
    
    # Ensure images have the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    # Determine the appropriate `win_size`
    min_dim = min(img1.shape[:2])  # Smallest spatial dimension
    win_size = min(7, min_dim) if min_dim >= 7 else min_dim  # Adjust dynamically
    
    return ssim(img1, img2, data_range=img1.max() - img1.min(), win_size=win_size, channel_axis=-1)

def train():
    config_path = "../configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    dataloader = get_dataloader(config_path)
    
    optimizer_g = Adam(generator.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
    loss_fn = GANLoss()
    
    num_epochs = config['train']['num_epochs']
    for epoch in range(num_epochs):
        epoch_g_loss, epoch_d_loss = 0.0, 0.0
        psnr_vals, ssim_vals = [], []
        
        for i, (real_imgs, target_imgs) in enumerate(dataloader):

            if i == 10:  # Stop after the first 10 images
                break

            real_imgs, target_imgs = real_imgs.to(device), target_imgs.to(device)
            
            # Train Discriminator
            optimizer_d.zero_grad()
            real_validity = discriminator(target_imgs)
            fake_imgs = generator(real_imgs).detach()
            fake_validity = discriminator(fake_imgs)
            d_loss = loss_fn.discriminator_loss(real_validity, fake_validity)
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            fake_imgs = generator(real_imgs)
            fake_validity = discriminator(fake_imgs)
            g_loss = loss_fn.generator_loss(fake_validity, fake_imgs, target_imgs)
            g_loss.backward()
            optimizer_g.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            # Calculate PSNR and SSIM
            for j in range(fake_imgs.shape[0]):
                fake_img = fake_imgs[j].permute(1, 2, 0).detach().cpu()
                target_img = target_imgs[j].permute(1, 2, 0).detach().cpu()
                
                psnr_val = calculate_psnr(fake_img, target_img)
                ssim_val = calculate_ssim(fake_img, target_img)
                
                psnr_vals.append(psnr_val)
                ssim_vals.append(ssim_val)
        
        epoch_psnr = sum(psnr_vals) / len(psnr_vals)
        epoch_ssim = sum(ssim_vals) / len(ssim_vals)
        
        print(f"Epoch {epoch + 1}/{num_epochs} | G Loss: {epoch_g_loss:.4f} | D Loss: {epoch_d_loss:.4f} | PSNR: {epoch_psnr:.2f} | SSIM: {epoch_ssim:.4f}")
    
if __name__ == '__main__':
    train()
