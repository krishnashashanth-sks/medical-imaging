import tensorflow as tf
from tensorflow import keras
import numpy as np
import os 
from tokenizers import BertWordPieceTokenizer, Tokenizer # Import Tokenizer for from_file
import os
from utils import create_pretraining_examples,mask_tokens,prepare_example
from model import BioBERTPretrainer
from losses import masked_lm_loss
from train import train_model

CORPUS_PATH = "medical_corpus"

# In a real scenario, this directory would contain your actual medical text files.
if not os.path.exists(CORPUS_PATH):
    os.makedirs(CORPUS_PATH)
    with open(os.path.join(CORPUS_PATH, "sample_medical_text.txt"), "w") as f:
        f.write("The patient presented with symptoms of acute myocardial infarction. Electrocardiogram showed ST-segment elevation. Treatment included aspirin, clopidogrel, and heparin. \n")
        f.write("Further diagnostic tests revealed coronary artery disease. Surgical intervention was deemed necessary. Recovery was prolonged but successful. \n")

# Define special tokens as required by the problem statement and BERT architecture
SPECIAL_TOKENS = {
    'pad_token': '[PAD]',
    'unk_token': '[UNK]',
    'cls_token': '[CLS]',
    'sep_token': '[SEP]',
    'mask_token': '[MASK]'
}

special_tokens_list = list(SPECIAL_TOKENS.values())

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=True,
    lowercase=True,
)

# Train the tokenizer
# We'll use the dummy corpus files for training
# In a real scenario, this would be a large collection of medical text files.
text_files = [os.path.join(CORPUS_PATH, f) for f in os.listdir(CORPUS_PATH) if f.endswith('.txt')]

print(f"Training tokenizer on {len(text_files)} files from {CORPUS_PATH}...")
tokenizer.train(
    files=text_files,
    vocab_size=30000, # A common vocabulary size for BERT-like models
    min_frequency=2,
    show_progress=True,
    special_tokens=special_tokens_list,
)

# Save the tokenizer as a single JSON file
TOKENIZER_PATH = "medical_wordpiece_tokenizer.json"
tokenizer.save(TOKENIZER_PATH) # Use tokenizer.save() for a single JSON file
print(f"Tokenizer trained and saved to {TOKENIZER_PATH}")

# Load the tokenizer from the single JSON file
tokenizer = Tokenizer.from_file(TOKENIZER_PATH) # Use Tokenizer.from_file()

# Set vocab_size based on the trained tokenizer
vocab_size = tokenizer.get_vocab_size()

print(f"Tokenizer loaded. Vocabulary size: {vocab_size}")

# Verify special tokens are correctly mapped
print(f"CLS Token ID: {tokenizer.token_to_id(SPECIAL_TOKENS['cls_token'])}")
print(f"SEP Token ID: {tokenizer.token_to_id(SPECIAL_TOKENS['sep_token'])}")
print(f"PAD Token ID: {tokenizer.token_to_id(SPECIAL_TOKENS['pad_token'])}")
print(f"MASK Token ID: {tokenizer.token_to_id(SPECIAL_TOKENS['mask_token'])}")
print(f"UNK Token ID: {tokenizer.token_to_id(SPECIAL_TOKENS['unk_token'])}")

MAX_SEQ_LEN = 128 # Define a maximum sequence length for our BioBERT model
MASK_PROB = 0.15 # Probability of masking a token for MLM

# Get special token IDs from the tokenizer using the SPECIAL_TOKENS dictionary
# Assuming the tokenizer has been loaded and is available in the environment
CLS_TOKEN_ID = tokenizer.token_to_id(SPECIAL_TOKENS['cls_token'])
SEP_TOKEN_ID = tokenizer.token_to_id(SPECIAL_TOKENS['sep_token'])
PAD_TOKEN_ID = tokenizer.token_to_id(SPECIAL_TOKENS['pad_token'])
MASK_TOKEN_ID = tokenizer.token_to_id(SPECIAL_TOKENS['mask_token'])
UNK_TOKEN_ID = tokenizer.token_to_id(SPECIAL_TOKENS['unk_token'])

# --- Load corpus sentences from text_files ---
corpus_sentences = []
for file_path in text_files:
    with open(file_path, "r", encoding="utf-8") as f:
        # Simple split by newline for demonstration. More robust splitting might be needed for real corpus.
        sentences = [line.strip() for line in f.readlines() if line.strip()]
        corpus_sentences.extend(sentences)

print(f"Loaded {len(corpus_sentences)} sentences from the medical corpus.")

# ---  Integrate and generate pre-training examples ---
raw_pretraining_examples = create_pretraining_examples(
    corpus_sentences, tokenizer, MAX_SEQ_LEN,
    CLS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID, MASK_TOKEN_ID
)

pretraining_dataset_elements = []
print("Applying MLM masking to examples...")
for example in raw_pretraining_examples:
    input_ids = example["input_ids"]
    masked_input_ids, mlm_labels = mask_tokens(
        input_ids, vocab_size, MASK_TOKEN_ID, CLS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID
    )

    pretraining_dataset_elements.append({
        "input_ids": masked_input_ids,
        "segment_ids": example["segment_ids"],
        "attention_mask": example["attention_mask"],
        "mlm_labels": mlm_labels,
        "nsp_labels": example["nsp_label"],
    })
print(f"Finished applying MLM masking. Total pre-training elements: {len(pretraining_dataset_elements)}")

# ---  Convert to tf.data.Dataset ---
# Define the output signature for the dataset
output_signature = (
    {
        "input_ids": tf.TensorSpec(shape=(MAX_SEQ_LEN,), dtype=tf.int32),
        "segment_ids": tf.TensorSpec(shape=(MAX_SEQ_LEN,), dtype=tf.int32),
        "attention_mask": tf.TensorSpec(shape=(MAX_SEQ_LEN,), dtype=tf.int32),
    },
    {
        "mlm_labels": tf.TensorSpec(shape=(MAX_SEQ_LEN,), dtype=tf.int32),
        "nsp_labels": tf.TensorSpec(shape=(), dtype=tf.int32),
    }
)

# Create the tf.data.Dataset
pretraining_dataset = tf.data.Dataset.from_generator(
    lambda: (prepare_example(ex) for ex in pretraining_dataset_elements), # Use a generator expression with prepare_example
    output_signature=output_signature
).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

# Model Parameters
EMBEDDING_DIM = 768
NUM_LAYERS = 12 # Number of Transformer blocks
NUM_HEADS = 12  # Number of attention heads
FF_DIM = 3072   # Feed-forward network dimension (usually 4 * EMBEDDING_DIM)
DROPOUT_RATE = 0.1

print(f"Model Configuration: EMBEDDING_DIM={EMBEDDING_DIM}, NUM_LAYERS={NUM_LAYERS}, NUM_HEADS={NUM_HEADS}, FF_DIM={FF_DIM}")

#  Instantiate the BioBERTPretrainer model
biobert_pretrainer = BioBERTPretrainer(
    max_seq_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    dropout_rate=DROPOUT_RATE
)


#  Implement the Next Sentence Prediction (NSP) loss function
nsp_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False) # NSPHead uses sigmoid activation, so from_logits=False

# Define metrics
mlm_metrics = [keras.metrics.SparseCategoricalAccuracy(name='mlm_accuracy')]
nsp_metrics = [keras.metrics.BinaryAccuracy(name='nsp_accuracy')]
mlm_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='mlm_accuracy')
nsp_accuracy = tf.keras.metrics.BinaryAccuracy(name='nsp_accuracy')

#  Create an Adam optimizer with a suitable learning rate
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#  Compile the BioBERTPretrainer model
biobert_pretrainer.compile(
    optimizer=optimizer,
    loss={
        "mlm_output": masked_lm_loss,
        "nsp_output": nsp_loss
    },
    metrics={
        "mlm_output": mlm_metrics,
        "nsp_output": nsp_metrics
    }
)

# Training
#  Define the number of epochs and steps per epoch
EPOCHS = 1 # For demonstration, set to a small number. In reality, this would be much higher (e.g., 5-10).
BATCH_SIZE = 32 # This should match the batch size used when creating pretraining_dataset

# Calculate steps per epoch based on the dataset size and batch size
NUM_PRETRAINING_EXAMPLES = len(pretraining_dataset_elements)
STEPS_PER_EPOCH = NUM_PRETRAINING_EXAMPLES // BATCH_SIZE
# Handle case where STEPS_PER_EPOCH is 0 due to small dataset (e.g., single batch)
if STEPS_PER_EPOCH == 0 and NUM_PRETRAINING_EXAMPLES > 0:
    STEPS_PER_EPOCH = 1

print(f"Starting pre-training for {EPOCHS} epochs.")
print(f"Total pre-training examples: {NUM_PRETRAINING_EXAMPLES}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Steps per epoch: {STEPS_PER_EPOCH}")

train_model(EPOCHS,pretraining_dataset,STEPS_PER_EPOCH,biobert_pretrainer,masked_lm_loss,nsp_loss,optimizer,mlm_accuracy,nsp_accuracy)

# We'll use a single example for demonstration
inference_batch_size = 1
dummy_inference_token_ids = tf.random.uniform(
    shape=(inference_batch_size, MAX_SEQ_LEN), minval=0, maxval=vocab_size, dtype=tf.int32
)
dummy_inference_segment_ids = tf.random.uniform(
    shape=(inference_batch_size, MAX_SEQ_LEN), minval=0, maxval=1, dtype=tf.int32
)
dummy_inference_attention_mask = tf.constant(
    np.ones((inference_batch_size, MAX_SEQ_LEN)), dtype=tf.int32
) # All 1s for a simple inference mask

print("--- Performing Inference ---")
# Perform inference
inference_outputs = biobert_pretrainer(
    input_ids=dummy_inference_token_ids,
    segment_ids=dummy_inference_segment_ids,
    attention_mask=dummy_inference_attention_mask,
    training=False # Set to False for inference mode
)

mlm_inference_logits = inference_outputs["mlm_output"]
nsp_inference_predictions = inference_outputs["nsp_output"]

print(f"Inference MLM logits shape: {mlm_inference_logits.shape}")
print(f"Inference NSP predictions shape: {nsp_inference_predictions.shape}")