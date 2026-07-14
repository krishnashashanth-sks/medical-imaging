import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from model import create_deepdta_model
from utils import preprocess_data

drug_chars = ['#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '=', '8', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'X', 'Z', '[', ']', '_', 'a', 'c', 'b', 'e', 'g', 'i', 'm', 'n', 'o', 'p', 's', 'r', 'u']
target_chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

#  SETUP CONSTANTS
DRUG_VOCAB_SIZE = len(drug_chars)
TARGET_VOCAB_SIZE = len(target_chars)
MAX_DRUG_LEN = 100
MAX_TARGET_LEN = 1000

# 3. DOWNLOAD AND PROCESS DATA
davis_y_url = "https://raw.githubusercontent.com/hkmztrk/deepdta/master/data/davis/Y"
drug_smiles_url = "https://raw.githubusercontent.com/hkmztrk/deepdta/master/data/davis/ligands_can.txt"
target_seq_url = "https://raw.githubusercontent.com/hkmztrk/deepdta/master/data/davis/proteins.txt"

try:
    print("Downloading Davis dataset...")
    y_res = requests.get(davis_y_url)
    y_res.raise_for_status()
    affinity_matrix = np.loadtxt(StringIO(y_res.text))

    smiles_list = requests.get(drug_smiles_url).text.splitlines()
    sequence_list = requests.get(target_seq_url).text.splitlines()

    flat_smiles, flat_targets, flat_affinities = [], [], []

    for i in range(len(sequence_list)):
        for j in range(len(smiles_list)):
            val = affinity_matrix[i, j]
            # Convert Kd to pKd: -log10(Kd/1e9)
            pkd = -np.log10(val / 1e9)
            flat_smiles.append(smiles_list[j])
            flat_targets.append(sequence_list[i])
            flat_affinities.append(pkd)

    # DEFINING THE MISSING VARIABLES HERE
    davis_drugs_encoded, davis_targets_encoded, davis_affinities = preprocess_data(flat_smiles, flat_targets, flat_affinities)

    # 4. PERFORM THE SPLIT
    X_train_drugs, X_test_drugs, X_train_targets, X_test_targets, train_affinities,test_affinities = train_test_split(
        davis_drugs_encoded, davis_targets_encoded, davis_affinities, test_size=0.2, random_state=42
    )

    print("\nSUCCESS! All variables defined and data split complete.")
    print(f"Train size: {len(X_train_drugs)}, Test size: {len(X_test_drugs)}")

except Exception as e:
    print(f"An error occurred: {e}","So, creating a dummy drug data")
    # Define parameters for dummy data generation
    NUM_TRAIN_SAMPLES = 1000
    NUM_VAL_SAMPLES = 200
    NUM_TEST_SAMPLES = 200

    # Generate dummy drug data (one-hot encoded style)
    X_train_drugs = np.random.rand(NUM_TRAIN_SAMPLES, MAX_DRUG_LEN, DRUG_VOCAB_SIZE).astype(np.float32)
    X_test_drugs = np.random.rand(NUM_TEST_SAMPLES, MAX_DRUG_LEN, DRUG_VOCAB_SIZE).astype(np.float32)

    # Generate dummy target data (one-hot encoded style)
    X_train_targets = np.random.rand(NUM_TRAIN_SAMPLES, MAX_TARGET_LEN, TARGET_VOCAB_SIZE).astype(np.float32)
    X_test_targets = np.random.rand(NUM_TEST_SAMPLES, MAX_TARGET_LEN, TARGET_VOCAB_SIZE).astype(np.float32)

    # Generate dummy affinity values (regression task)
    train_affinities = np.random.rand(NUM_TRAIN_SAMPLES).astype(np.float32) * 10 # Example range for affinities
    test_affinities = np.random.rand(NUM_TEST_SAMPLES).astype(np.float32) * 10


# Define training parameters
BATCH_SIZE = 128
EPOCHS = 20 # You can increase this for better convergence

# Instantiate the model
deepdta_model = create_deepdta_model(
    MAX_DRUG_LEN, DRUG_VOCAB_SIZE, MAX_TARGET_LEN, TARGET_VOCAB_SIZE
)

# Compile the model
deepdta_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("DeepDTA model architecture created and compiled.")
print(deepdta_model.summary())

print("Starting model training...")

# Train the model
history = deepdta_model.fit(
    {'drug_input': X_train_drugs, 'target_input': X_train_targets},
    train_affinities,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=1
)

print("\nModel training complete.")

print("\nEvaluating the model on the final test set...")

# Evaluate the model on the final test set
loss, mae = deepdta_model.evaluate(
    {'drug_input': X_test_drugs, 'target_input': X_test_targets},
    test_affinities,
    verbose=0
)

print(f"Final Test Loss (MSE): {loss:.4f}")
print(f"Final Test MAE: {mae:.4f}")

print("\nPerforming inference (predictions) on a small subset of the test data...")

# Predict on a small subset of the final test data (e.g., first 10 samples)
num_predictions = 10
sample_drugs = X_test_drugs[:num_predictions]
sample_targets = X_test_targets[:num_predictions]
sample_true_affinities = test_affinities[:num_predictions]

predicted_affinities = deepdta_model.predict(
    {'drug_input': sample_drugs, 'target_input': sample_targets}
).flatten()

print("\n--- Sample Predictions ---")
for i in range(num_predictions):
    print(f"Sample {i+1}: True Affinity = {sample_true_affinities[i]:.4f}, Predicted Affinity = {predicted_affinities[i]:.4f}")
