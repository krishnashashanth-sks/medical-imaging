from tensorflow.keras import layers
from layers import TokenAndPositionEmbedding,TransformerBlock
from tensorflow import keras
import tensorflow as tf

def create_med_palm_like_model(
    vocab_size,
    maxlen,
    embed_dim,
    num_heads,
    ff_dim,
    num_layers,
    dropout_rate
):
    # Input layers
    input_ids = layers.Input(shape=(maxlen,), dtype=tf.int32, name="input_ids")
    segment_ids = layers.Input(shape=(maxlen,), dtype=tf.int32, name="segment_ids")
    # attention_mask is usually created internally or can be passed for padding handling
    # For simplicity, we'll assume dense attention here. For padded sequences, it would be needed.

    # Create embeddings
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer((input_ids, segment_ids))

    # Stack Transformer blocks
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x) # `training` handled by default in TransformerBlock.call

    # --- Pre-training Heads ---

    # Masked Language Model (MLM) head
    # This head predicts the masked tokens. It usually takes the output for all tokens.
    mlm_output = layers.Dense(vocab_size, name="mlm_output")(x)

    # Next Sentence Prediction (NSP) head
    # This head takes the [CLS] token representation (first token) and predicts if the sentences are consecutive.
    cls_token_output = layers.Lambda(lambda x: x[:, 0, :])(x) # Extract [CLS] token representation
    nsp_output = layers.Dense(1, activation="sigmoid", name="nsp_output")(cls_token_output)

    # Create the model
    model = keras.Model(inputs=[input_ids, segment_ids], outputs=[mlm_output, nsp_output])

    return model
