# model/tokenizer.py

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string-to-index
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # index-to-string
        self.vocab_size = len(self.chars)

    def encode(self, text):
        """Encode string to list of token IDs."""
        return [self.stoi[c] for c in text]

    def decode(self, indices):
        """Decode list of token IDs back to string."""
        return ''.join([self.itos[i] for i in indices])
