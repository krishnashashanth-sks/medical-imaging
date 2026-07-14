import numpy as np
from scipy.ndimage import zoom

def preprocess_volume(volume,target_shape=(64,64,64),normalize=True):
  current_shape=volume.shape
  zoom_factors=[ts/cs for ts,cs in zip(target_shape,current_shape)]
  resampled_volume=zoom(volume,zoom_factors,order=1)
  if normalize:
    min_val,max_val=resampled_volume.min(),resampled_volume.max()
    if max_val>min_val:
      resampled_volume=(resampled_volume-min_val)/(max_val-min_val)
    else:
      resampled_volume=np.zeros_like(resampled_volume)
  return resampled_volume

def preprocess_medmnist_data(dataset, target_shape=(64, 64, 64)):
    images = []
    labels = []
    for i in range(len(dataset)):
        img, lab = dataset[i]
        # MedMNIST images are typically (D, H, W) for n_channels=1, but sometimes might come as (D, H, W, 1).
        # Squeeze to ensure it's 3D before passing to preprocess_volume.
        processed_img = preprocess_volume(img.squeeze().astype(np.float32), target_shape=target_shape, normalize=True)

        # Convert 4D multi-class label to a 3D binary mask (any non-background organ as foreground)
        # Assuming '0' is background class. `(lab > 0).any(axis=-1)` checks if any class channel (excluding background implicitly) is active.
        binary_lab_3d = (lab > 0).any(axis=-1).astype(np.float32)

        # Handle the case where binary_lab_3d might become a scalar (ndim=0)
        if binary_lab_3d.ndim == 0:
            # If it's a scalar, create a 3D array of the target shape filled with that scalar value
            processed_lab = np.full(target_shape, binary_lab_3d.item(), dtype=np.uint8)
        else:
            processed_lab = preprocess_volume(binary_lab_3d, target_shape=target_shape, normalize=False)
            # Threshold labels back to binary after resampling, if needed (interpolation can create floats)
            processed_lab = (processed_lab > 0.5).astype(np.uint8)

        images.append(processed_img)
        labels.append(processed_lab)
    return np.array(images), np.array(labels)

# Placeholder for training data generation
def data_generator(images, masks, batch_size):
    num_samples = len(images)
    while True:
        # Shuffle data for each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            batch_images = np.array([images[i] for i in batch_indices])
            # Cast masks to float32 to match model output for loss function
            batch_masks = np.array([masks[i] for i in batch_indices]).astype(np.float32)
            # Add channel dimension if not present (e.g., (batch_size, D, H, W, 1))
            yield np.expand_dims(batch_images, -1), np.expand_dims(batch_masks, -1)