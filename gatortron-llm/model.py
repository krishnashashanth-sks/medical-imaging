from layers import Embeddings,Decoder,Generator
import torch.nn as nn

def make_model(vocab_size, num_layers, d_model, d_ff, dropout):
    """Helper function: Construct a model from hyperparameters."""
    # Create a dummy MHA layer, we will replace this later with a full implementation
    # For now, SelfAttention is used in TransformerBlock, and it's already defined with h=8

    # Instantiate Embeddings
    embed = Embeddings(vocab_size, d_model, dropout)

    # Instantiate Decoder with stacked TransformerBlocks
    decoder = Decoder(num_layers, d_model, dropout)

    # Create the final linear output layer (Generator)
    generator = Generator(d_model, vocab_size)

    # Assemble the model
    model = nn.Sequential(embed, decoder, generator)

    # Initialize model parameters with Xavier uniform initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print("Model constructed and parameters initialized.")
    return model

print("Generator class and make_model function defined.")