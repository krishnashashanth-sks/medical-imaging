import numpy as np

def model_inference(model, test_images, threshold=0.5):
    """
    Performs inference using the trained V-Net model and post-processes predictions.

    Args:
        model (tf.keras.Model): The trained V-Net model.
        test_images (np.ndarray): A NumPy array of test images. Expected shape (num_samples, D, H, W).
        threshold (float): Threshold to binarize the sigmoid output.

    Returns:
        np.ndarray: A NumPy array of binary predicted segmentation masks.
    """
    # Add channel dimension if not present and batch dimension
    # Model expects (batch_size, D, H, W, channels)
    input_images = np.expand_dims(test_images, axis=-1) # Add channel dim

    print(f"Performing inference on {input_images.shape[0]} images...")
    predictions = model.predict(input_images)
    print("Inference complete.")

    # Post-process predictions: binarize based on threshold
    binary_predictions = (predictions > threshold).astype(np.uint8)

    # Remove the channel dimension if only one channel was used
    return binary_predictions.squeeze(axis=-1)
