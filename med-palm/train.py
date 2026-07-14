import tensorflow as tf

@tf.function
def train_step(input_ids, segment_ids, mlm_labels, nsp_labels,med_palm_like_model,mlm_loss_fn,nsp_loss_fn,optimizer,mlm_accuracy_metric,nsp_accuracy_metric):
    with tf.GradientTape() as tape:
        mlm_output, nsp_output = med_palm_like_model([input_ids, segment_ids], training=True)

        # Calculate MLM loss
        # Create a sample weight mask for MLM labels, ignoring -100
        mlm_sample_weight = tf.cast(tf.not_equal(mlm_labels, -100), tf.float32)
        mlm_loss = mlm_loss_fn(mlm_labels, mlm_output, sample_weight=mlm_sample_weight)

        # Calculate NSP loss
        nsp_loss = nsp_loss_fn(nsp_labels, nsp_output)

        total_loss = mlm_loss + nsp_loss # You might want to weigh these losses

    gradients = tape.gradient(total_loss, med_palm_like_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, med_palm_like_model.trainable_variables))

    # Update metrics
    mlm_accuracy_metric.update_state(mlm_labels, mlm_output, sample_weight=mlm_sample_weight)
    nsp_accuracy_metric.update_state(nsp_labels, nsp_output)

    return total_loss
