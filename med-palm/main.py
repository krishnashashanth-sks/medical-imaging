from train import train_step
from model import create_med_palm_like_model
import tensorflow as tf

EMBEDDING_DIM = 768
NUM_LAYERS = 12 # Number of Transformer blocks
NUM_HEADS = 12  # Number of attention heads
FF_DIM = 3072   # Feed-forward network dimension (usually 4 * EMBEDDING_DIM)
DROPOUT_RATE = 0.1
MAX_SEQ_LEN = 128 
VOCAB_SIZE=1000

print(f"Model Configuration: EMBEDDING_DIM={EMBEDDING_DIM}, NUM_LAYERS={NUM_LAYERS}, NUM_HEADS={NUM_HEADS}, FF_DIM={FF_DIM}")

# Instantiate the model
med_palm_like_model = create_med_palm_like_model(
    vocab_size=VOCAB_SIZE,
    maxlen=MAX_SEQ_LEN,
    embed_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    dropout_rate=DROPOUT_RATE
)
med_palm_like_model.summary()

print("Simplified Med-PALM 2-like model created.")

LEARNING_RATE = 0.0001 # Defined LEARNING_RATE based on kernel state
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Loss for Masked Language Model (MLM)
# We use from_logits=True because the MLM output does not have an activation function yet
mlm_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Loss for Next Sentence Prediction (NSP)
nsp_loss_fn = tf.keras.losses.BinaryCrossentropy()

# Define metrics
mlm_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="mlm_accuracy")
nsp_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name="nsp_accuracy")

sample_input =(
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


total_loss=train_step(sample_input['input_ids'], sample_input['segment_ids'], sample_input['mlm_labels'], sample_input['nsp_labels'],med_palm_like_model,mlm_loss_fn,nsp_loss_fn,optimizer,mlm_accuracy_metric,nsp_accuracy_metric)

print("Total loss:",total_loss.numpy())

# Prepare input for inference (add batch dimension)
inference_input_ids = tf.constant(sample_input['input_ids'], dtype=tf.int32)[tf.newaxis, :]
inference_segment_ids = tf.constant(sample_input['segment_ids'], dtype=tf.int32)[tf.newaxis, :]

# Perform inference
mlm_predictions, nsp_prediction = med_palm_like_model(
    [inference_input_ids, inference_segment_ids], training=False
)

print("\n--- Inference Results ---")
print("MLM Output Shape:", mlm_predictions.shape)
print("NSP Output:", nsp_prediction.numpy())

# To get the predicted token for MLM, we can take the argmax of the logits
predicted_mlm_tokens = tf.argmax(mlm_predictions, axis=-1).numpy()
print("Predicted MLM Tokens (first 10):", predicted_mlm_tokens[0, :10])

# For NSP, a value > 0.5 typically indicates 'IsNextSentence'
print("NSP Probability (IsNextSentence):", nsp_prediction.numpy()[0][0])
if nsp_prediction.numpy()[0][0] > 0.5:
    print("Predicted: Is Next Sentence")
else:
    print("Predicted: Not Next Sentence")