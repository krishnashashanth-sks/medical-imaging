from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import pandas as pd

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, num_classes=14):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.num_classes = num_classes
        # Ensure labels are numeric. You might need to preprocess your CSV
        # to convert '1.0' to 1, '0.0' to 0, and '-1.0' (uncertain) to 0 or treat as missing.
        # For simplicity, assuming your CSV already has binary labels (0 or 1).
        self.labels = self.annotations.iloc[:, 1:].values # Assuming first column is image path, rest are labels

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.annotations.iloc[idx, 0]) # Assuming image path in first column
        image = Image.open(img_name).convert('RGB') # Ensure 3 channels
        labels = torch.tensor(self.labels[idx].astype(float), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

