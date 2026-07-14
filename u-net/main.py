import matplotlib.pyplot as plt
import torch.optim as optim
from dataset import MedMNISTSegmentationDataset
from model import UNet
import torch
from torch.utils.data import DataLoader
import medmnist
import os
from medmnist import INFO
import torchvision.transforms as transforms
import torch.nn as nn
from train import train_model

in_channels = 3 # Changed to 3 for RGB MedMNIST PathMNIST images
num_classes = 1 # Changed from 2 to 1 for binary segmentation with BCEWithLogitsLoss

model = UNet(in_channels=in_channels, num_classes=num_classes)

data_flag = 'pathmnist'
download = True

# Get dataset info
info = INFO[data_flag]
n_channels = info['n_channels'] # This will be 3 for PathMNIST
n_classes = len(info['label']) # Number of classification labels

IMG_SIZE = (96, 96, 96)

# Define transforms for MedMNIST to be compatible with 3D SwinUNETR
# MedMNIST images (PathMNIST) are 3 channels (RGB) but 2D
medmnist_image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE[1], IMG_SIZE[2])), # Resize H, W
    transforms.ToTensor(), # Convert PIL Image to Tensor (C, H, W)
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]), # Normalize for 3 channels
    # Add a depth dimension of 1 for SwinUNETR (C, 1, H, W)
    transforms.Lambda(lambda x: x.unsqueeze(1)) # Add depth dimension
])

# For the masks, PathMNIST provides classification labels, not segmentation masks.
# We'll create a dummy mask from the label. This transform prepares it to match image dims.
medmnist_mask_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE[1], IMG_SIZE[2]), interpolation=transforms.InterpolationMode.NEAREST), # Resize H, W
    transforms.ToTensor(), # Convert to Tensor (1, H, W)
    # Add a depth dimension of 1 for SwinUNETR (1, 1, H, W)
    transforms.Lambda(lambda x: x.unsqueeze(1)) # Add depth dimension
])

# Ensure the root directory exists
os.makedirs('./data', exist_ok=True)

# Load MedMNIST PathMNIST dataset (raw dataset from medmnist library)
PathMNIST = medmnist.PathMNIST(root='./data', download=download, as_rgb=True, split='train')

# Instantiate our custom wrapper dataset for SwinUNETR
train_dataset = MedMNISTSegmentationDataset(
    medmnist_dataset=PathMNIST, # Use the raw medmnist dataset
    image_transform=medmnist_image_transform,
    mask_transform=medmnist_mask_transform,
    target_img_size=IMG_SIZE # Pass the target 3D size
)

# Create a DataLoader
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to the device
model.to(device)

# Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss() # Good for binary segmentation where model output is raw logits
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 5 # You can adjust this

train_model(num_epochs,model,train_dataloader,optimizer,criterion,device)

print("Training setup complete: model, loss function, and optimizer initialized.")

# Set model to evaluation mode
model.eval()

# Get one batch from the training dataloader for visualization
# Using next(iter(dataloader)) is a common way to get a single batch
images, true_masks = next(iter(train_dataloader))

# Move images and masks to the device
images = images.to(device)
true_masks = true_masks.to(device)

# Perform inference without gradient calculation
with torch.no_grad():
    outputs = model(images)

# Apply sigmoid to get probabilities (since BCEWithLogitsLoss was used)
predicted_masks_probs = torch.sigmoid(outputs)

# Convert probabilities to binary masks (e.g., threshold at 0.5)
predicted_masks = (predicted_masks_probs > 0.5).float()

# Move data back to CPU for visualization with matplotlib
images_cpu = images.cpu().numpy()
true_masks_cpu = true_masks.cpu().numpy()
predicted_masks_cpu = predicted_masks.cpu().numpy()

print(f"Shape of images for visualization: {images_cpu.shape}")
print(f"Shape of true masks for visualization: {true_masks_cpu.shape}")
print(f"Shape of predicted masks for visualization: {predicted_masks_cpu.shape}")

# Visualize a few samples
num_samples_to_display = 10# Display up to 4 images from the batch

plt.figure(figsize=(15, num_samples_to_display * 5))
for i in range(min(num_samples_to_display, images_cpu.shape[0])):
    # Original Image
    plt.subplot(num_samples_to_display, 3, i * 3 + 1)
    # MedMNIST images are RGB, so we need to permute dimensions if it's [C, H, W] to [H, W, C]
    # Also, imshow expects values in [0, 1] or [0, 255], our normalized images are [-1, 1]
    # Convert back to [0, 1] for display: (image * 0.5 + 0.5)
    display_image = images_cpu[i].transpose(1, 2, 0) * 0.5 + 0.5
    plt.imshow(display_image)
    plt.title(f'Original Image {i+1}')
    plt.axis('off')

    # Ground Truth Mask
    plt.subplot(num_samples_to_display, 3, i * 3 + 2)
    # Masks are [1, H, W], so squeeze the channel dimension
    plt.imshow(true_masks_cpu[i].squeeze(), cmap='gray')
    plt.title(f'Ground Truth Mask {i+1}')
    plt.axis('off')

    # Predicted Mask
    plt.subplot(num_samples_to_display, 3, i * 3 + 3)
    plt.imshow(predicted_masks_cpu[i].squeeze(), cmap='gray')
    plt.title(f'Predicted Mask {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()