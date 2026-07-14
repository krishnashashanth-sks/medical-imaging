from evaluate import evaluate_model
from inference import model_inference
from losses import hybrid_loss,dice_coef,dice_loss
from model import VNet
from utils import preprocess_medmnist_data,data_generator
# pip install medmnist
import medmnist
from medmnist import INFO
import numpy as np 
from visualize import visualize_segmentation

model = VNet()
model.compile(optimizer='adam', loss=hybrid_loss, metrics=[dice_coef, 'accuracy'])

# Define dimensions for the dummy data
dummy_shape = (32, 32, 32)

# --- MedMNIST Data Loading and Preprocessing ---
data_flag = 'organmnist3d' # Corrected key to lowercase
download = True

info = INFO[data_flag]
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# Load Data
# Use a small subset for demonstration if full dataset is too large or slow
train_dataset = DataClass(split='train', download=download)
val_dataset = DataClass(split='val', download=download)
test_dataset = DataClass(split='test', download=download)

print("Preprocessing training data...")
X_train, y_train = preprocess_medmnist_data(train_dataset)
print("Preprocessing validation data...")
X_val, y_val = preprocess_medmnist_data(val_dataset)
print("Preprocessing test data...")
X_test, y_test = preprocess_medmnist_data(test_dataset)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Example of model training
model = VNet()
model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coef])

train_generator = data_generator(X_train, y_train, batch_size=2)
val_generator = data_generator(X_val, y_val, batch_size=2)

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 2,
    epochs=50, # Adjust number of epochs
    validation_data=val_generator,
    validation_steps=len(X_val) // 2
)

predicted_masks = model_inference(model, X_test[:5]) # Predict on first 5 test images for example

print(f"Shape of predicted masks: {predicted_masks.shape}")
print(f"Unique values in first predicted mask: {np.unique(predicted_masks[0])}")


# Visualize a few examples
num_samples_to_visualize = 3

print(f"Visualizing segmentation for {num_samples_to_visualize} samples from the test set...")

for i in range(num_samples_to_visualize):
    # Select a sample (image, true mask, predicted mask)
    sample_image = X_test[i]
    sample_true_mask = y_test[i]
    sample_predicted_mask = predicted_masks[i]

    # You can specify a slice_idx or let it default to the middle slice
    visualize_segmentation(sample_image, sample_true_mask, sample_predicted_mask, slice_idx=32)

print("Visualization complete.")
print("Starting model evaluation...")
evaluate_model(model, X_test, y_test)
print("Model evaluation complete.")