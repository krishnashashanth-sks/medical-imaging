from model import AdvancedSegResNet
from utils import generate_dummy_data,dice_coefficient
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf

# Define parameters
IMG_SIZE = (128, 128) # Example image size
NUM_CHANNELS = 1     # e.g., grayscale MRI/CT slices
NUM_CLASSES = 2      # e.g., background and one foreground class
BATCH_SIZE = 4

# Generate some dummy data
num_train_samples = 100
num_val_samples = 20

train_images, train_masks = generate_dummy_data(num_train_samples, IMG_SIZE, NUM_CHANNELS, NUM_CLASSES)
val_images, val_masks = generate_dummy_data(num_val_samples, IMG_SIZE, NUM_CHANNELS, NUM_CLASSES)

print(f"Train images shape: {train_images.shape}")
print(f"Train masks shape: {train_masks.shape}")

# Convert to TensorFlow Datasets (recommended for larger datasets)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


input_shape = (IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS)
model = AdvancedSegResNet(input_shape, num_classes=NUM_CLASSES, filters=(32, 64, 128, 256)) # Using 4 levels for example

model.summary()

# Define loss function
if NUM_CLASSES == 1:
    loss_fn = keras.losses.BinaryCrossentropy()
else:
    loss_fn = keras.losses.CategoricalCrossentropy()


# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=loss_fn,
              metrics=[dice_coefficient, 'accuracy'])

# Train the model
print("\nStarting mod el training...")
history = model.fit(
    train_dataset,
    epochs=10, # Adjust number of epochs as needed
    validation_data=val_dataset
)

print("\nModel training finished.")

# Take one sample from the validation set for inference
sample_image = val_images[0]
sample_mask = val_masks[0]

# Add a batch dimension for prediction
input_image_for_prediction = np.expand_dims(sample_image, axis=0)

# Make prediction
predicted_mask = model.predict(input_image_for_prediction)[0]

# Post-process predicted mask for visualization (e.g., if multi-class, get the argmax)
if NUM_CLASSES > 1:
    predicted_mask_display = np.argmax(predicted_mask, axis=-1)
    sample_mask_display = np.argmax(sample_mask, axis=-1)
else:
    predicted_mask_display = (predicted_mask > 0.5).astype(np.float32)[..., 0] # For binary, threshold at 0.5
    sample_mask_display = sample_mask[..., 0]

# Visualize the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(sample_image[..., 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('True Mask')
plt.imshow(sample_mask_display, cmap='viridis')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Predicted Mask')
plt.imshow(predicted_mask_display, cmap='viridis')
plt.axis('off')

plt.show()
