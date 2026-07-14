import torch

def generate_sequence(model, start_sequence, max_len, device, EOS_TOKEN=2): # Assuming EOS_TOKEN is 2, adjust if needed
    """Generates a sequence of token IDs from a trained model."""
    model.eval() # Set the model to evaluation mode

    # Convert start_sequence to a tensor and add batch dimension
    input_sequence = torch.tensor(start_sequence).unsqueeze(0).to(device)

    generated_sequence = start_sequence[:]

    with torch.no_grad(): # Disable gradient calculations
        for _ in range(max_len - len(start_sequence)): # Generate tokens until max_len is reached
            # Pass the current sequence through the model
            # Our model is nn.Sequential(embed, decoder, generator)
            # It expects the input for Embeddings, which is the current sequence of token IDs
            model_output = model(input_sequence) # Shape: (batch_size, sequence_length, vocab_size)

            # Extract logits for the last token in the sequence
            next_token_logits = model_output[:, -1, :]

            # Sample the next token (using argmax for simplicity)
            predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()

            # Check for EOS token
            if predicted_token_id == EOS_TOKEN:
                break

            # Append the predicted token to the generated sequence list
            generated_sequence.append(predicted_token_id)

            # Update the input_sequence tensor for the next iteration
            input_sequence = torch.cat([
                input_sequence,
                torch.tensor([[predicted_token_id]], device=device)
            ], dim=1)

    return generated_sequence