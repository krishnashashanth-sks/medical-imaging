import torchvision.transforms  as transforms
import torch
from model import DenseNet
import medmnist
from medmnist import INFO
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from train import train_model

# Define transformations for training and validation
IMG_SIZE = 224 # CheXNet typically uses 224x224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats, common for X-rays too
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example Usage (replace with your actual paths)
# csv_file_path = 'path/to/your/labels.csv'
# image_directory = 'path/to/your/images/'

# train_dataset = ChestXrayDataset(csv_file=csv_file_path, image_dir=image_directory, transform=train_transform, num_classes=14)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

#  Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#  Instantiate the model
# Default CheXNet parameters: growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, num_classes=14
model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, num_classes=14).to(device)

#  Define Loss function and Optimizer
criterion = nn.BCEWithLogitsLoss() # Suitable for multi-label classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Common choice for deep learning

# Install MedMNIST if not already installed
# pip install medmnist

# Load ChestMNIST dataset using MedMNIST library
data_flag = 'chestmnist'
download = True # Set to True to download the dataset if not present

info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# Training dataset and loader
train_dataset = DataClass(split='train', transform=train_transform, download=download)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)

# Validation dataset and loader (using 'test' split for validation)
val_dataset = DataClass(split='test', transform=val_transform, download=download)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=2)

print(f"ChestMNIST dataset '{data_flag}' loaded.")
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}, Label shape: {train_dataset[0][1].shape}")

# Training Loop Parameters
num_epochs = 10 # You can adjust this
print_every_n_batches = 100 # Print loss every N batches

train_model(num_epochs,model,train_loader,val_loader,optimizer,criterion,print_every_n_batches,device)

print("\n--- Performing Model Inference ---")

# Ensure the model is in evaluation mode
model.eval()

# Get a single batch from the validation loader for demonstration
# You can also load a new image and apply val_transform to it
with torch.no_grad():
    dataiter = iter(val_loader)
    images, labels = next(dataiter)

    # Take the first image from the batch
    single_image = images[0].unsqueeze(0).to(device) # Add batch dimension and move to device
    true_labels = labels[0].cpu().numpy()

    # Forward pass
    outputs = model(single_image)

    # Apply sigmoid to get probabilities (since BCEWithLogitsLoss was used)
    probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()

    print("Input image shape:", single_image.shape)
    print("True labels (binary):")
    print(true_labels)
    print("\nPredicted probabilities for 14 classes:")

    # Get class names from MedMNIST info if available, otherwise use generic names
    if 'label' in info and isinstance(info['label'], dict):
        class_names = list(info['label'].values())
    else:
        class_names = [f'Class_{i+1}' for i in range(len(probabilities))]

    for i, prob in enumerate(probabilities):
        print(f"  {class_names[i]}: {prob:.4f}")

print("Inference complete for a single sample.")
# To perform inference on multiple images, you would iterate through your DataLoader
# or batch your custom images and process them similarly.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.transforms as T
import seaborn as sns # Import seaborn

# Denormalize the image for display
# We need the mean and std used in the original normalization
# Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
inv_normalize = T.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# single_image is already on 'device' (GPU), move it to CPU and remove batch dimension
img_to_display = single_image.cpu().squeeze(0)
img_to_display = inv_normalize(img_to_display) # Denormalize
img_to_display = img_to_display.permute(1, 2, 0).numpy() # Convert from C, H, W to H, W, C for matplotlib
img_to_display = np.clip(img_to_display, 0, 1) # Clip values to [0, 1]

# Create a DataFrame for easy plotting (same as before)
results_df = pd.DataFrame({
    'Class': class_names,
    'Predicted Probability': probabilities,
    'True Label': true_labels
})

# Melt the DataFrame for seaborn barplot
melted_df = results_df.melt(id_vars='Class', var_name='Type', value_name='Value')

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 2]})

# Plot the image
axes[0].imshow(img_to_display)
axes[0].set_title('Input Image')
axes[0].axis('off') # Hide axes for image

# Plot the bar chart
sns.barplot(x='Class', y='Value', hue='Type', data=melted_df, palette={'Predicted Probability': 'skyblue', 'True Label': 'lightcoral'}, ax=axes[1])
axes[1].set_title('Predicted Probabilities vs. True Labels')
axes[1].set_xlabel('Pathology Class')
axes[1].set_ylabel('Value (Probability / Binary Label)')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Metric')

plt.tight_layout()
plt.show()