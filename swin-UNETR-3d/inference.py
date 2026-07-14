import torch
import numpy as np

def inference_swinunetr(model, data_loader, device, n_channels):
    """
    Performs inference using the SwinUNETR model.

    Args:
        model (nn.Module): The trained SwinUNETR model.
        data_loader (DataLoader): DataLoader for the inference dataset.
        device (torch.device): The device (cpu or cuda) to run inference on.
        n_channels (int): Number of input channels for the images.

    Returns:
        tuple: A tuple containing concatenated NumPy arrays of:
            - predicted_masks (np.array): All predicted masks.
            - true_masks (np.array): All true masks.
    """
    model.eval()  # Set the model to evaluation mode
    all_predicted_masks = []
    all_true_masks = []

    with torch.no_grad():  # Disable gradient calculations during inference
        for batch_idx, (images, masks) in enumerate(data_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Ensure input has the correct number of channels for the model.
            if images.shape[1] != n_channels:
                print(f"Warning: Input channels mismatch. Expected {n_channels}, got {images.shape[1]}. Slicing to {n_channels} channels.")
                images = images[:, :n_channels, :, :, :]

            outputs = model(images)

            # SwinUNETR model returns a list of outputs if deep supervision is enabled.
            # For inference, we typically take the highest resolution output, which is outputs[0].
            if isinstance(outputs, list):
                predicted_logits = outputs[0]  # Take the highest resolution output
            else:
                predicted_logits = outputs

            # Apply argmax to get predicted class labels (for multi-class segmentation)
            # If it's binary segmentation (out_channels=1), apply sigmoid and threshold.
            if model.output_conv.out_channels == 1:
                predicted_probs = torch.sigmoid(predicted_logits)
                predicted_binary_masks = (predicted_probs > 0.5).float()
                all_predicted_masks.append(predicted_binary_masks.cpu().numpy())
            else:
                predicted_labels = torch.argmax(predicted_logits, dim=1)
                all_predicted_masks.append(predicted_labels.cpu().numpy())

            all_true_masks.append(masks.cpu().numpy())

    # Concatenate all predictions and true masks from batches
    predicted_masks_np = np.concatenate(all_predicted_masks, axis=0)
    true_masks_np = np.concatenate(all_true_masks, axis=0)

    print("Inference complete.")
    return predicted_masks_np, true_masks_np
