import torch
import torch.nn.functional as F
import math
import random

def attention(query, key, value, mask=None, dropout=None):
  """Computes scaled dot-product attention."""
  d_k = query.size(-1)
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)

  p_attn = F.softmax(scores, dim=-1) # Apply softmax

  if dropout is not None:
    p_attn = dropout(p_attn) # Apply dropout after softmax

  return torch.matmul(p_attn, value), p_attn
  
# Dummy data generator for demonstration
class DataIterator:
    def __init__(self, vocab_size, max_len, batch_size, num_batches,device):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.current_batch = 0
        self.device=device
    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch < self.num_batches:
            self.current_batch += 1
            # Simulate source and target sequences
            src_len = random.randint(1, self.max_len)
            tgt_len = random.randint(1, self.max_len)

            src = torch.randint(1, self.vocab_size, (self.batch_size, src_len)).to(self.device)
            # Target input sequence (decoder input)
            tgt_input = torch.randint(1, self.vocab_size, (self.batch_size, tgt_len)).to(self.device)
            # Target output sequence (labels for loss calculation)
            # Shifted right, so the last token of tgt_input is not in tgt_output
            tgt_output = torch.randint(1, self.vocab_size, (self.batch_size, tgt_len)).to(self.device)
            return src, tgt_input, tgt_output
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches
