import numpy as np
import tensorflow as tf

def generate_dummy_data(num_samples, img_size, num_channels, num_classes):
    """Generates dummy medical image-like data for demonstration."""
    images = np.random.rand(num_samples, img_size[0], img_size[1], num_channels).astype(np.float32)
    masks = np.random.randint(0, num_classes, size=(num_samples, img_size[0], img_size[1], 1)).astype(np.float32)
    if num_classes > 1: # One-hot encode for multi-class segmentation
        masks = tf.keras.utils.to_categorical(masks, num_classes=num_classes)
    return images, masks

# Define metrics (e.g., Dice Coefficient or IoU)
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / (denominator + tf.keras.backend.epsilon())
