import os
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import load_medmnist_pathmnist
from generator import make_generator_model
from discriminator import make_discriminator_model
from train import train
from losses import *

# Define hyperparameters
IMG_HEIGHT = 32
IMG_WIDTH = 32 
IMG_CHANNELS = 3 # Changed to 3 for RGB medical images like PathMNIST
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
LATENT_DIM = 100 # Dimension of the latent space (noise vector)
NUM_GENERATORS = 3 # Number of diverse generators in MADGAN
BATCH_SIZE = 16 # Reduced from 32 to decrease RAM usage
BUFFER_SIZE = 500 # Reduced from 1000 to decrease RAM usage for shuffling
EPOCHS = 5000

G_LR = 1e-4 # Generator learning rate
D_LR = 1e-4 # Discriminator learning rate
BETA_1 = 0.5 # Adam optimizer beta1 parameter

# Loss weights (example, these need tuning for diversity loss)
LAMBDA_DIVERSITY = 1.0 # Weight for diversity loss

os.makedirs('./data',exist_ok=True)

# Load MedMNIST PathMNIST data
real_images = load_medmnist_pathmnist()

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices(real_images)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generators = [make_generator_model(LATENT_DIM,IMG_CHANNELS,IMG_HEIGHT,IMG_WIDTH) for _ in range(NUM_GENERATORS)]

discriminator = make_discriminator_model(IMG_SHAPE)

feature_extractor = tf.keras.Model(
    inputs=discriminator.input,
    outputs=discriminator.layers[4].output # Assuming layer 4 is the desired feature layer
)
# Optimizers for each component
generator_optimizers = [tf.keras.optimizers.Adam(learning_rate=G_LR, beta_1=BETA_1) for _ in range(NUM_GENERATORS)]
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LR, beta_1=BETA_1)

# Explicitly build the discriminator optimizer with the discriminator's variables
discriminator_optimizer.build(discriminator.trainable_variables)

print("Starting MADGAN training...")

train(dataset, EPOCHS,generators,discriminator,generator_loss,discriminator_loss,diversity_loss,generator_optimizers,discriminator_optimizer,BATCH_SIZE,LATENT_DIM,NUM_GENERATORS,LAMBDA_DIVERSITY)

# Perform Model Inference

# Generate noise vectors for inference
num_examples_to_generate = 10 # You can change this number
inference_noise_vectors = [tf.random.normal([num_examples_to_generate, LATENT_DIM]) for _ in range(NUM_GENERATORS)]

print(f"Generating {num_examples_to_generate} images from each of the {NUM_GENERATORS} generators...")

# Function to denormalize images from [-1, 1] to [0, 1] for display
def denormalize_img(img):
    return (img * 0.5) + 0.5

plt.figure(figsize=(num_examples_to_generate * 2, NUM_GENERATORS * 2))
plt.suptitle("Generated Images from MADGAN Generators", fontsize=16)

for gen_idx in range(NUM_GENERATORS):
    # Generate images from the current generator
    generated_images = generators[gen_idx](inference_noise_vectors[gen_idx], training=False)

    for i in range(num_examples_to_generate):
        plt.subplot(NUM_GENERATORS, num_examples_to_generate, gen_idx * num_examples_to_generate + i + 1)
        plt.imshow(denormalize_img(generated_images[i, :, :, :]))
        plt.axis('off')
        if i == 0:
            plt.title(f'Gen {gen_idx+1}', loc='left')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
