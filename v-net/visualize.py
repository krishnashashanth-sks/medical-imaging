import matplotlib.pyplot as plt

def visualize_segmentation(image, true_mask, pred_mask, slice_idx=None):
    """
    Visualizes an image, its true segmentation mask, and predicted mask.

    Args:
        image (np.ndarray): The input 3D image (D, H, W).
        true_mask (np.ndarray): The ground truth 3D mask (D, H, W).
        pred_mask (np.ndarray): The predicted 3D mask (D, H, W).
        slice_idx (int, optional): The 3D slice index to visualize. If None, uses the middle slice.
    """
    if slice_idx is None:
        slice_idx = image.shape[0] // 2 # Use the middle slice

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image[slice_idx, :, :], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(true_mask[slice_idx, :, :], cmap='viridis')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(pred_mask[slice_idx, :, :], cmap='magma')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.suptitle(f'Slice {slice_idx} of 3D Volume')
    plt.show()