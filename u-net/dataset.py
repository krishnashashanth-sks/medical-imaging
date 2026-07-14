from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class MedMNISTSegmentationDataset(Dataset):
    def __init__(self, medmnist_dataset, image_transform=None, mask_transform=None, target_img_size=(96,96,96)):
        self.medmnist_dataset = medmnist_dataset
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.target_img_size = target_img_size

    def __len__(self):
        return len(self.medmnist_dataset)

    def __getitem__(self, idx):
        img, label = self.medmnist_dataset[idx]

        # MedMNIST images are PIL Image. Apply image_transform.
        if self.image_transform:
            img = self.image_transform(img)

        # For segmentation, we need a pixel-wise mask. PathMNIST gives a classification label.
        # Let's create a *dummy* binary mask based on the label for demonstration purposes.
        # E.g., if label is > 0, make the mask all ones; otherwise, all zeros.
        # In a real scenario, this would be loaded from a mask file.
        # The label comes as a tensor, usually [0] for the class index.
        dummy_mask_val = 1.0 if label.item() > 0 else 0.0

        # Create a blank PIL image for the dummy mask with the target 2D size
        dummy_mask = Image.fromarray(np.full((self.target_img_size[1], self.target_img_size[2]), int(dummy_mask_val * 255), dtype=np.uint8))

        if self.mask_transform:
            dummy_mask = self.mask_transform(dummy_mask)

        # Ensure mask is float
        dummy_mask = dummy_mask.float() # Ensure it's float for BCEWithLogitsLoss

        return img, dummy_mask