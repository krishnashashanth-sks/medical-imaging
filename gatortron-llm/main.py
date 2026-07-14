import torch.nn as nn
from model import make_model
import torch
from scheduler import NoamOpt
from utils import DataIterator
from train import run_epoch
from losses import SimpleLossCompute
from inference import generate_sequence

criterion = nn.CrossEntropyLoss(ignore_index=0) # Assuming 0 is the padding index

VOCAB_SIZE = 10000  # Example vocabulary size
NUM_LAYERS = 6      # Example number of Transformer blocks
D_MODEL = 512       # Example model dimension
D_FF = 2048         # Example feed-forward network dimension
DROPOUT = 0.1       # Example dropout rate

# Instantiate the model before defining the optimizer
model = make_model(VOCAB_SIZE, NUM_LAYERS, D_MODEL, D_FF, DROPOUT)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

# Instantiate the scheduler
# Using D_MODEL for model_size and a common warmup value, e.g., 2000 steps
scheduler = NoamOpt(optimizer, D_MODEL, warmup=2000)

# Detect if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to the chosen device
model.to(device)

epochs = 10 # Define the number of training epochs

# Main training loop
print("Starting training...")
for epoch in range(epochs):
    model.train() # Set model to training mode
    # Instantiate a new DataIterator for each epoch if you want different data each time
    # Or, if data is small, can iterate over the same data multiple times
    train_data_iterator = DataIterator(VOCAB_SIZE, max_len=10, batch_size=32, num_batches=100,device=device)

    print(f"\nEpoch {epoch + 1}/{epochs}")
    avg_loss = run_epoch(train_data_iterator, model, SimpleLossCompute(criterion, scheduler), device)
    print(f"End of Epoch {epoch + 1}, Average Loss: {avg_loss:.2f}")

print("Training complete.")

start_sequence = [1, 5, 2] # Example: a sequence of token IDs
max_generation_length = 20 # Maximum length for the generated sequence

print(f"Starting sequence: {start_sequence}")

generated_tokens = generate_sequence(model, start_sequence, max_generation_length, device, EOS_TOKEN=2)

print(f"Generated sequence: {generated_tokens}")
print("Model inference demonstration complete.")