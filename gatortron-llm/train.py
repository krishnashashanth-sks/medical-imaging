import time

def run_epoch(data_iter, model, loss_compute, device):
    """Standard training and logging function for an epoch."""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        src, tgt_input, tgt_output = batch
        src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

        # Perform forward pass
        # The `make_model` function assembles Embeddings, Decoder, and Generator
        # So, model(src, tgt_input) would represent the full forward pass for a sequence-to-sequence model
        # However, our current model is more like a decoder-only LLM.
        # The model's forward method from `make_model` should be updated if it's meant to take both.
        # For now, let's assume `model` directly takes the input that produces target_output
        # We need to refine the `model`'s forward signature in `make_model` or assume the `Generator` takes the last decoder output.

        # For a decoder-only model like GatorTron, we usually feed the target input and predict the next token.
        # The current `make_model` uses nn.Sequential(embed, decoder, generator).
        # If `embed` takes `tgt_input`, `decoder` processes it, and `generator` predicts.
        # So, the input to the model should be `tgt_input`.
        # The `TransformerBlock` and `Decoder` as implemented don't take a mask, which is crucial for causality.
        # This needs to be addressed for a true LLM.

        # Assuming model takes tgt_input and produces logits for next token prediction
        out = model(tgt_input) # Assuming model processes tgt_input for next token prediction

        # Calculate loss and perform backward pass
        loss = loss_compute(out, tgt_output)
        total_loss += loss
        total_tokens += (tgt_output != 0).sum().item() # Assuming 0 is padding
        tokens += (tgt_output != 0).sum().item()

        if i % 50 == 1:
            elapsed = time.time() - start
            print(f"Epoch Step: {i} Loss: {loss / (tgt_output != 0).sum().item():.2f} Tokens per Sec: {tokens / elapsed:.2f}")
            start = time.time()
            tokens = 0

    return total_loss / total_tokens