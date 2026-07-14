import tensorflow as tf

# 3. Implement custom Masked Language Modeling (MLM) loss function
def masked_lm_loss(real_labels, pred_logits):
    # Only calculate loss for positions where real_labels is not -100 (which indicates non-masked tokens)
    loss_mask = tf.math.not_equal(real_labels, -100)

    # Flatten the inputs for sparse_softmax_cross_entropy_with_logits
    real_labels_flat = tf.reshape(real_labels[loss_mask], [-1])
    pred_logits_flat = tf.reshape(pred_logits[loss_mask], [-1, tf.shape(pred_logits)[-1]])

    if tf.size(real_labels_flat) == 0: # Handle case where no tokens are masked in a batch
        return tf.constant(0.0)

    sparse_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=real_labels_flat,
        logits=pred_logits_flat
    )
    return tf.reduce_mean(sparse_loss)
