import torch.nn as nn
import torch
from utils import attention
import math
import torch.nn.functional as F

class LayerNorm(nn.Module):
  def __init__(self,features,eps=1e-6):
    super(LayerNorm,self).__init__()
    self.a_2=nn.Parameter(torch.ones(features))
    self.b_2=nn.Parameter(torch.zeros(features))
    self.eps=eps
  def forward(self,x):
    mean=x.mean(-1,keepdim=True)
    std=x.std(-1,keepdim=True)
    return self.a_2*(x-mean)/(std+self.eps)+self.b_2
    
class SublayerConnection(nn.Module):
  def __init__(self,size,dropout):
    super(SublayerConnection,self).__init__()
    self.norm=LayerNorm(size)
    self.dropout=nn.Dropout(dropout)
  def forward(self,x,sublayer):
    return x+self.dropout(sublayer(self.norm(x)))

class SelfAttention(nn.Module):
  """Multi-Head Attention module."""
  def __init__(self, d_model, h, dropout=0.1):
    super(SelfAttention, self).__init__()
    assert d_model % h == 0
    self.d_k = d_model // h
    self.h = h
    # Four linear layers: for query, key, value, and final output projection
    self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query_input, key_input, value_input, mask=None):
    """Applies multi-head attention."""
    if mask is not None:
      # Same mask applies to all h heads
      mask = mask.unsqueeze(1)

    nbatches = query_input.size(0)

    # 1) Do all the linear projections in batch from d_model => h x d_k
    query, key, value = \
        [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        for l, x in zip(self.linears, (query_input, key_input, value_input))]

    # 2) Apply attention on all the projected vectors in batch.
    x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

    # 3) "Concat" using a view and apply a final linear.
    x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
    return self.linears[-1](x)

class Swish(nn.Module):
    """Implements the Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForwardNetwork(nn.Module):
    """Implements a position-wise feed-forward network with SwiGLU activation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.swish = Swish()

    def forward(self, x):
        # SwiGLU logic: (Swish(w_gate(x)) * w_up(x)) -> w_down -> dropout
        return self.dropout(self.w_down(self.swish(self.w_gate(x)) * self.w_up(x)))

class PositionalEncoding(nn.Module):
    """Implements positional encoding for input sequences."""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

class Embeddings(nn.Module):
    """Combines token embeddings and positional embeddings."""
    def __init__(self, vocab_size, d_model, dropout, max_len=5000):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)  # Lookup table for token embeddings
        self.pe = PositionalEncoding(d_model, dropout, max_len) # Positional Encoding
        self.d_model = d_model

    def forward(self, x):
        # Scale the token embeddings by sqrt(d_model) before adding positional encodings
        return self.pe(self.lut(x) * math.sqrt(self.d_model))

class TransformerBlock(nn.Module):
    """A single Transformer block, composed of self-attention and feed-forward network."""
    def __init__(self, size, dropout):
        super(TransformerBlock, self).__init__()
        self.self_attn = SelfAttention(size, h=8, dropout=dropout) # Assuming h=8 for multi-head attention
        self.feed_forward = FeedForwardNetwork(size, size * 4, dropout) # Use the fully implemented FFN
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, x):
        """Applies the Transformer block operations."""
        # Self-attention takes query, key, value, and mask. For typical decoder block, these are all x
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer[1](x, self.feed_forward) # Apply feed-forward with residual connection and layer norm
        return x

class Decoder(nn.Module):
    """Core decoder is a stack of N TransformerBlocks."""
    def __init__(self, num_layers, size, dropout):
        super(Decoder, self).__init__()
        # Create num_layers instances of TransformerBlock
        self.layers = nn.ModuleList([TransformerBlock(size, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(size) # LayerNorm at the end of the stack
        self.size = size # Store size for the LayerNorm

    def forward(self, x):
        """Pass the input through each TransformerBlock in turn."""
        for layer in self.layers:
            x = layer(x) # TransformerBlock.forward(x) does not take mask based on its current definition in 76c2b895
        return self.norm(x)

class Generator(nn.Module):
    """Defines the standard linear + softmax generation step."""
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)