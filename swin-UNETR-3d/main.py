import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import SwinUNETR
from dataset import MedMNISTSegmentationDataset
import matplotlib.pyplot as plt
import numpy as np
from train import train_model
from inference import inference_swinunetr

SWINUNETR_IMG_SIZE = (96, 96, 96) # Re-defining for robustness
img_size = SWINUNETR_IMG_SIZE

data_flag = 'pathmnist' # Assuming data_flag is defined in previous cells
info = INFO[data_flag] # Assuming INFO is imported from medmnist
in_channels = info['n_channels'] # PathMNIST has 3 channels
out_channels = 1 # For binary segmentation example

# Instantiate the SwinUNETR model
model = SwinUNETR(
    img_size=img_size,
    in_channels=in_channels,
    out_channels=out_channels,
    feature_size=24, # Initial feature dimension
    depths=(2, 2, 2, 2), # Number of blocks in each encoder/decoder stage
    num_heads=(3, 6, 12, 24), # Number of attention heads in each stage
    window_size=(7, 7, 7), # Window size for 3D attention
    drop_path_rate=0.1
)

# Move model to device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Ensure the root directory exists for MedMNIST data
os.makedirs('./data', exist_ok=True)

# Re-define data_flag and download if not globally available, for robustness
data_flag = 'pathmnist'
download = True

# Get MedMNIST dataset info
info = INFO[data_flag]

# Define target size for SwinUNETR
SWINUNETR_IMG_SIZE = (96, 96, 96)

# Define transforms for MedMNIST to be compatible with 3D SwinUNETR
medmnist_swinunetr_image_transform = transforms.Compose([
    transforms.Resize((SWINUNETR_IMG_SIZE[1], SWINUNETR_IMG_SIZE[2])), # Resize H, W
    transforms.ToTensor(), # Convert PIL Image to Tensor (C, H, W)
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]), # Normalize for 3 channels
    # Add a depth dimension of 1, then repeat it to match SWINUNETR_IMG_SIZE[0]
    transforms.Lambda(lambda x: x.unsqueeze(1).repeat(1, SWINUNETR_IMG_SIZE[0], 1, 1)) # C, D, H, W
])

medmnist_swinunetr_mask_transform = transforms.Compose([
    transforms.Resize((SWINUNETR_IMG_SIZE[1], SWINUNETR_IMG_SIZE[2]), interpolation=transforms.InterpolationMode.NEAREST), # Resize H, W
    transforms.ToTensor(), # Convert to Tensor (1, H, W)
    # Add a depth dimension of 1, then repeat it to match SWINUNETR_IMG_SIZE[0]
    transforms.Lambda(lambda x: x.unsqueeze(1).repeat(1, SWINUNETR_IMG_SIZE[0], 1, 1)) # 1, D, H, W
])

# Load MedMNIST PathMNIST dataset (raw dataset from medmnist library)
PathMNIST = medmnist.PathMNIST(root='./data', download=download, as_rgb=True, split='train')

# Instantiate our custom wrapper dataset for SwinUNETR
train_dataset = MedMNISTSegmentationDataset(
    medmnist_dataset=PathMNIST, # Use the raw medmnist dataset
    image_transform=medmnist_swinunetr_image_transform,
    mask_transform=medmnist_swinunetr_mask_transform,
    target_img_size=SWINUNETR_IMG_SIZE # Pass the target 3D size
)

# Create a DataLoader
batch_size = 2 # Reduced batch size to manage GPU memory
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss() # Good for binary segmentation where model output is raw logits
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 5 # You can adjust this

train_model(num_epochs,model,train_dataloader,optimizer,criterion,device)

print("\n--- Performing SwinUNETR Model Inference ---")

# Run the inference function
predicted_masks_np, true_masks_np = inference_swinunetr(model, train_dataloader, device, in_channels)

print(f"Shape of all predicted masks: {predicted_masks_np.shape}")
print(f"Shape of all true masks: {true_masks_np.shape}")
print("SwinUNETR Inference complete.")

# --- Visualization of Inference Results ---

# Get a few samples for visualization
num_samples_to_visualize = 3

plt.figure(figsize=(15, num_samples_to_visualize * 5))

# Iterate through the first few samples to display
for i in range(min(num_samples_to_visualize, predicted_masks_np.shape[0])):
    # For 3D data, select a central slice for visualization
    depth_slice_idx = predicted_masks_np.shape[2] // 2

    # Get the original image (need to get it from the dataloader again or store it during inference)
    # For simplicity, let's grab directly from the dataloader for now, assuming order is maintained.
    # A more robust approach would be to return original images from inference_swinunetr if needed.
    # For now, let's extract images from the first few batches of the dataloader.
    current_image_tensor = None
    for batch_idx, (images, _) in enumerate(train_dataloader):
        if batch_idx * train_dataloader.batch_size <= i:
            if (batch_idx + 1) * train_dataloader.batch_size > i:
                current_image_tensor = images[i % train_dataloader.batch_size]
                break
    
    if current_image_tensor is None:
        print(f"Could not retrieve original image for sample {i}, skipping visualization for this sample.")
        continue

    display_image = current_image_tensor.cpu().numpy() # (C, D, H, W)
    display_image = (display_image * 0.5 + 0.5) # Denormalize from [-1, 1] to [0, 1]
    
    # Select a slice for display (e.g., first channel, middle depth slice)
    # PathMNIST is RGB, so `display_image[0]` might not be suitable, need to average or pick a channel if desired.
    # For simplicity, let's take a single channel (e.g., first) and the middle depth slice
    img_slice = display_image[0, depth_slice_idx, :, :]

    # Ground Truth Mask (squeeze channel dim if present, take middle depth slice)
    true_mask_slice = true_masks_np[i].squeeze()[depth_slice_idx, :, :]

    # Predicted Mask (squeeze channel dim if present, take middle depth slice)
    predicted_mask_slice = predicted_masks_np[i].squeeze()[depth_slice_idx, :, :]

    plt.subplot(num_samples_to_visualize, 3, i * 3 + 1)
    plt.imshow(img_slice, cmap='gray') # Using grayscale for single channel image slice
    plt.title(f'Original Image {i+1}\n(Slice {depth_slice_idx})')
    plt.axis('off')

    plt.subplot(num_samples_to_visualize, 3, i * 3 + 2)
    plt.imshow(true_mask_slice, cmap='viridis')
    plt.title(f'Ground Truth Mask {i+1}\n(Slice {depth_slice_idx})')
    plt.axis('off')

    plt.subplot(num_samples_to_visualize, 3, i * 3 + 3)
    plt.imshow(predicted_mask_slice, cmap='magma')
    plt.title(f'Predicted Mask {i+1}\n(Slice {depth_slice_idx})')
    plt.axis('off')

plt.tight_layout()
plt.show()
