import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from layers import TokenAndPositionalEmbedding,TransformerBlock,MLMHead,NSPHead

class BioBERTPretrainer(keras.Model):
  def __init__(self,max_seq_len,vocab_size,embedding_dim,num_layers,num_heads,ff_dim,dropout_rate=0.1,type_vocab_size=2,**kwargs):
    super(BioBERTPretrainer,self).__init__(**kwargs)
    self.max_seq_len=max_seq_len
    self.vocab_size=vocab_size
    self.embedding_dim=embedding_dim
    self.num_layers=num_layers
    self.num_heads=num_heads
    self.ff_dim=ff_dim
    self.dropout_rate=dropout_rate
    self.type_vocab_size=type_vocab_size
    self.token_position_embedding=TokenAndPositionalEmbedding(
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        type_vocab_size=type_vocab_size
    )
    self.encoder_blocks=[
        TransformerBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            rate=dropout_rate
        )
        for _ in range(num_layers)
    ]
    self.pooler=layers.Dense(embedding_dim,activation='tanh',name='pooler_dense')
    self.mlm_head=MLMHead(vocab_size=vocab_size,embedding_dim=embedding_dim)
    self.nsp_head=NSPHead()
  def call(self,input_ids,segment_ids,attention_mask,training=False):
    embeddings=self.token_position_embedding([input_ids,segment_ids])

    # Prepare attention mask for Transformer blocks:
    # MultiHeadSelfAttention expects a mask where 1 indicates padding and 0 indicates valid tokens.
    # Our input `attention_mask` is 1 for valid tokens and 0 for padding. So we invert it.
    processed_attention_mask = tf.cast(tf.math.equal(attention_mask, 0), tf.float32)

    encoder_output=embeddings
    for encoder_block in self.encoder_blocks:
      encoder_output=encoder_block(encoder_output,training=training,mask=processed_attention_mask)

    # Take the [CLS] token's output for NSP
    first_token_output=encoder_output[:,0,:]
    pooled_output=self.pooler(first_token_output)

    mlm_logits=self.mlm_head(encoder_output, embeddings=self.token_position_embedding.token_embeddings.weights[0]) # Pass token embeddings for weight sharing
    nsp_predictions=self.nsp_head(pooled_output)

    # Return outputs as a dictionary, mapping to the label names in the tf.data.Dataset
    return {"mlm_output": mlm_logits, "nsp_output": nsp_predictions}

  def get_config(self):
        config = super(BioBERTPretrainer, self).get_config()
        config.update(
            {
                "max_seq_len": self.max_seq_len,
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "type_vocab_size": self.type_vocab_size
            }
        )
        return config
