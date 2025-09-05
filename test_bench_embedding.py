# --- Embedding mini demo ---

import torch
import torch.nn as nn

# 1) Pretend we already have a tokenizer with vocab_size=49
vocab_size = 49

# 2) Make a small batch of token IDs (batch=2, ctx=5)
#    Think of these as coming from your get_batch() function
X = torch.tensor([
    [23, 24, 25, 1,  0],   # row 0
    [ 3, 10,  7, 8, 15],   # row 1
], dtype=torch.long)

print("X (token ids) shape:", X.shape)
print(X)

# 3) Create an embedding layer (like your model's token_embedding)
embed_dim = 16
token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

# 4) Convert token IDs -> vectors of floats
E = token_embedding(X)
print("\nE (embedded vectors) shape:", E.shape)  # (batch, ctx, embed_dim)
print("E sample row 0, pos 0 vector:\n", E[0,0])

# 5) (Optional) Add positional embeddings like GPT does
pos_embedding = nn.Embedding(num_embeddings=X.shape[1], embedding_dim=embed_dim)
positions = torch.arange(0, X.shape[1]).unsqueeze(0).expand(X.shape[0], -1)  # shape (batch, ctx)
P = pos_embedding(positions)

# 6) Combine token + positional embeddings (elementwise add)
H = E + P
print("\nH (token + position) shape:", H.shape)
print("H[0,0] first 5 dims:", H[0,0,:5])

# 7) Show that these are floats and backprop works
#    (Fake a tiny linear head and a dummy loss)
head = nn.Linear(embed_dim, vocab_size)
logits = head(H)                              # (batch, ctx, vocab)
targets = torch.randint(0, vocab_size, X.shape)  # random targets just to test plumbing
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
print("\nDummy CE loss:", float(loss))

loss.backward()  # prove gradients flow
print("Grad on token_embedding.weight shape:", token_embedding.weight.grad.shape)
