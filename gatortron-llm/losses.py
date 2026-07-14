class SimpleLossCompute:
    """A simple loss compute and gradient update function."""
    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, normalize_batch_size=True):
        # x: model output (logits), y: true labels
        # Reshape x to (batch_size * sequence_length, vocab_size)
        # Reshape y to (batch_size * sequence_length)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))

        if normalize_batch_size:
            loss = loss / y.size(0) # Normalize by batch size

        if self.opt is not None:
            loss.backward()
            self.opt.step() # NoamOpt already calls optimizer.step()
            self.opt.optimizer.zero_grad() # Manually zero_grad from the underlying optimizer

        return loss.data.item() * y.size(0) # Return total loss for the batch

