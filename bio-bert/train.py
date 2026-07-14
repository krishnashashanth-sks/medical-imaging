import tensorflow as tf
import time

def train_model(epochs,pretraining_dataset,steps_per_epoch,biobert_pretrainer,masked_lm_loss,nsp_loss,optimizer,mlm_accuracy,nsp_accuracy):
    for epoch in range(epochs):
        start_time = time.time()

        # Reset metrics for the new epoch
        mlm_accuracy.reset_state()
        nsp_accuracy.reset_state()

        total_mlm_loss = 0.0
        total_nsp_loss = 0.0
        total_loss = 0.0

        for (batch_idx, (inputs, labels)) in enumerate(pretraining_dataset):
            if batch_idx >= steps_per_epoch: # Ensure we don't go over computed steps if dataset is small
                break

            # 7. Use tf.GradientTape to compute gradients
            with tf.GradientTape() as tape:
                # 4. Pass the input features to the biobert_pretrainer model
                # Unpack the 'inputs' dictionary into keyword arguments expected by the model's call method
                predictions = biobert_pretrainer(**inputs, training=True)
                mlm_pred_logits = predictions["mlm_output"]
                nsp_pred_logits = predictions["nsp_output"]

                # 5. Calculate MLM loss and NSP loss
                current_mlm_loss = masked_lm_loss(labels["mlm_labels"], mlm_pred_logits)
                current_nsp_loss = nsp_loss(labels["nsp_labels"], nsp_pred_logits)

                # 6. Combine the MLM and NSP losses
                current_total_loss = current_mlm_loss + current_nsp_loss

            # 8. Apply the computed gradients to the optimizer
            gradients = tape.gradient(current_total_loss, biobert_pretrainer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, biobert_pretrainer.trainable_variables))

            # Update metrics
            mlm_accuracy.update_state(labels["mlm_labels"], mlm_pred_logits)
            nsp_accuracy.update_state(labels["nsp_labels"], nsp_pred_logits)

            total_mlm_loss += current_mlm_loss
            total_nsp_loss += current_nsp_loss
            total_loss += current_total_loss

            # 10. Print or log the training progress periodically
            if (batch_idx + 1) % 1 == 0: # Log every 1 batch for small dataset
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{steps_per_epoch}: "
                    f"Total Loss: {total_loss / (batch_idx + 1):.4f}, "
                    f"MLM Loss: {total_mlm_loss / (batch_idx + 1):.4f}, "
                    f"NSP Loss: {total_nsp_loss / (batch_idx + 1):.4f}, "
                    f"MLM Accuracy: {mlm_accuracy.result():.4f}, "
                    f"NSP Accuracy: {nsp_accuracy.result():.4f}")

        end_time = time.time()
        print(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds.")
        print(f"Epoch {epoch+1} Metrics: "
            f"MLM Accuracy: {mlm_accuracy.result():.4f}, "
            f"NSP Accuracy: {nsp_accuracy.result():.4f}\n")

    print("Pre-training loop finished.")