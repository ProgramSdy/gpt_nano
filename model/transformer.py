# model/transformer.py

import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, context_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # (Optional) weight tying helps a bit for small models
        self.lm_head.weight = self.token_embedding.weight

    def _causal_mask(self, T, device):
        # shape (T, T); mask future positions with -inf
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, x):
        # x: (B, T) token ids
        B, T = x.shape
        assert T <= self.context_length, "sequence length exceeds context_length"

        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = self.token_embedding(x) + self.position_embedding(pos)

        # CAUSAL MASK so the model cannot look ahead
        src_mask = self._causal_mask(T, x.device)  # (T, T)

        h = self.encoder(h, mask=src_mask)
        h = self.ln_f(h)
        logits = self.lm_head(h)  # (B, T, vocab_size)
        return logits
