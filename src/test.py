import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.generator import Generator
import torchvision
import torch.nn.functional as F

def load_checkpoint(checkpoint_path, generator):
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    print(f"Checkpoint loaded: {checkpoint_path}")

def process_single_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Add batch dimension

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # Assuming images are normalized to [0, 1]
    return psnr.item()

def test_single_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    checkpoint_path = "../checkpoints/epoch_599.pth"
    load_checkpoint(checkpoint_path, generator)
    generator.eval()

    input_image_path = "../dataset/test/input/31.jpg"  
    output_image_path = "../outputs/enhanced_image.png"

    # Transform to convert image to tensor without resizing
    tran = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Process the real image
    real_image = process_single_image(input_image_path, tran, device)

    with torch.no_grad():
        # Process the image with the generator
        enhanced_image = generator(real_image)

    # Save the enhanced image as is, without resizing
    torchvision.utils.save_image(enhanced_image, output_image_path)
    print(f"Enhanced image saved at: {output_image_path}")

    # Ensure sizes match for PSNR calculation
    real_image_clamped = torch.clamp(real_image, 0, 1)
    enhanced_image_clamped = torch.clamp(enhanced_image, 0, 1)
    
    # Resize real_image_clamped to match enhanced_image_clamped size if needed
    if real_image_clamped.shape != enhanced_image_clamped.shape:
        real_image_clamped = F.interpolate(real_image_clamped, size=enhanced_image_clamped.shape[2:], mode='bilinear', align_corners=False)

    # PSNR Calculation
    psnr_value = calculate_psnr(real_image_clamped, enhanced_image_clamped)
    print(f"PSNR between the input and enhanced image: {psnr_value:.2f} dB")

if __name__ == "__main__":
    test_single_image()