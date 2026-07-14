import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) should be divisible by num_heads ({num_heads})")

        self.depth = embedding_dim // num_heads

        self.query_dense = layers.Dense(embedding_dim)
        self.key_dense = layers.Dense(embedding_dim)
        self.value_dense = layers.Dense(embedding_dim)

        self.final_dense = layers.Dense(embedding_dim)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        query, key, value = inputs[0], inputs[1], inputs[2]
        batch_size = tf.shape(query)[0]

        # 1. Apply projection layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. Reshape for multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. Calculate scaled dot-product attention
        # (batch_size, num_heads, seq_len_q, depth) * (batch_size, num_heads, depth, seq_len_k)
        # = (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention_logits = tf.matmul(query, key, transpose_b=True)
        scaled_attention_logits /= tf.math.sqrt(tf.cast(self.depth, tf.float32))

        # Apply mask if provided
        if mask is not None:
            # Add a large negative number to the masked positions so softmax outputs 0
            if len(mask.shape) == 2: # (batch_size, seq_len)
                # Expand to (batch_size, 1, 1, seq_len) for broadcasting across heads and query positions
                mask = mask[:, tf.newaxis, tf.newaxis, :]
            elif len(mask.shape) == 3: # (batch_size, seq_len_q, seq_len_k) for decoder attention
                # Expand to (batch_size, 1, seq_len_q, seq_len_k) for broadcasting across heads
                mask = mask[:, tf.newaxis, :, :]

            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # 4. Multiply attention weights by value
        # (batch_size, num_heads, seq_len_q, seq_len_k) * (batch_size, num_heads, seq_len_v, depth)
        # = (batch_size, num_heads, seq_len_q, depth)
        scaled_attention_output = tf.matmul(attention_weights, value)

        # 5. Concatenate outputs from all attention heads
        scaled_attention_output = tf.transpose(scaled_attention_output, perm=[0, 2, 1, 3]) # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention_output,
                                      (batch_size, -1, self.embedding_dim)) # (batch_size, seq_len_q, embedding_dim)

        # 6. Apply final output projection layer
        output = self.final_dense(concat_attention)

        return output, attention_weights
        
class TransformerBlock(layers.Layer):
    def __init__(self, embedding_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embedding_dim, num_heads)
        # Position-wise feed-forward network
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embedding_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training, mask=None):
        # Multi-head self-attention part
        attn_output, _ = self.att(inputs=[inputs, inputs, inputs], mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Add & Norm

        # Feed-forward network part
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Add & Norm

class TokenAndPositionalEmbedding(layers.Layer):
    def __init__(self, max_seq_len, vocab_size, embedding_dim, type_vocab_size=2, **kwargs):
        super(TokenAndPositionalEmbedding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.type_vocab_size = type_vocab_size

        # Token embeddings
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="token_embeddings"
        )
        # Segment embeddings (for A/B sentence types in NSP)
        self.segment_embeddings = layers.Embedding(
            input_dim=type_vocab_size,
            output_dim=embedding_dim,
            name="segment_embeddings"
        )
        # Positional embeddings
        self.position_embeddings = layers.Embedding(
            input_dim=max_seq_len,
            output_dim=embedding_dim,
            name="position_embeddings"
        )

    def call(self, inputs):
        token_ids, segment_ids = inputs

        sequence_length = tf.shape(token_ids)[-1]
        position_ids = tf.range(start=0, limit=sequence_length, delta=1)

        token_embedded = self.token_embeddings(token_ids)
        segment_embedded = self.segment_embeddings(segment_ids)
        position_embedded = self.position_embeddings(position_ids)

        # Sum token, segment, and positional embeddings
        combined_embeddings = token_embedded + segment_embedded + position_embedded
        return combined_embeddings

    def get_config(self):
        config = super(TokenAndPositionalEmbedding, self).get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "type_vocab_size": self.type_vocab_size,
        })
        return config


class MLMHead(layers.Layer):
  def __init__(self,vocab_size,embedding_dim,**kwargs):
    super(MLMHead,self).__init__(**kwargs)
    self.vocab_size=vocab_size
    self.embedding_dim=embedding_dim
    self.dense_projection=layers.Dense(embedding_dim,activation='gelu',name='mlm_dense_projection')
    self.layer_norm=layers.LayerNormalization(epsilon=1e-12,name='mlm_layer_norm')
    self.output_bias=self.add_weight(
        shape=(vocab_size,),
        initializer=tf.zeros_initializer(),
        trainable=True,
        name='mlm_output_bias'
    )
    # Removed self.decoder as a Dense layer to directly use weight sharing via matmul

  # build method is not strictly necessary if we perform matmul in call
  # def build(self,input_shape):
  #   super(MLMHead,self).build(input_shape)

  def call(self,inputs,embeddings=None):
    x=self.dense_projection(inputs)
    x=self.layer_norm(x)

    if embeddings is not None:
      # For weight sharing, directly perform matrix multiplication
      # 'embeddings' (from token_embeddings.weights[0]) has shape (vocab_size, embedding_dim)
      # We need to transpose it to (embedding_dim, vocab_size) to act as a kernel
      # 'x' has shape (batch_size, seq_len, embedding_dim)
      # Resulting logits will have shape (batch_size, seq_len, vocab_size)
      logits=tf.matmul(x, tf.transpose(embeddings))
    else:
      # If embeddings are not provided for weight sharing, raise an error
      raise ValueError("Embeddings must be provided to MLMHead for weight sharing.")

    logits=logits+self.output_bias
    return logits

  def get_config(self):
    config=super(MLMHead,self).get_config()
    config.update({
        "vocab_size":self.vocab_size,
        "embedding_dim":self.embedding_dim
    })
    return config

class NSPHead(layers.Layer):
  def __init__(self,**kwargs):
    super(NSPHead,self).__init__()
    self.dense=layers.Dense(1,activation='sigmoid',name='nsp_dense')
  def call(self,inputs):
    predictions=self.dense(inputs)
    return predictions
  def get_config(self):
    return super(NSPHead,self).get_config()