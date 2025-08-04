# train.py

import numpy as np
import torch
from torch import nn, optim
from model.transformer import TransformerModel
import pickle
from tqdm import tqdm  # 👈 import tqdm

# Load data
train_data = np.memmap("data/train.npy", dtype=np.uint16, mode='r')
val_data = np.memmap("data/val.npy", dtype=np.uint16, mode='r')

# Convert to PyTorch tensors
train_data = torch.from_numpy(train_data).long()
val_data = torch.from_numpy(val_data).long()

# Load tokenizer to get vocab size
with open("data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = tokenizer.vocab_size

# Model hyperparameters
embed_dim = 128
num_heads = 4
num_layers = 2
context_length = 128
batch_size = 64
learning_rate = 3e-4
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = TransformerModel(vocab_size, embed_dim, num_heads, num_layers, context_length).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Helper: generate a batch of input-target pairs
def get_batch(data, batch_size, context_length):
    ix = torch.randint(len(data) - context_length - 1, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    return x.to(device), y.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    print(f"\n🚀 Starting Epoch {epoch+1}/{num_epochs}")
    for step in tqdm(range(100), desc=f"Epoch {epoch+1}", leave=False):  # 👈 wrap tqdm
        x_batch, y_batch = get_batch(train_data, batch_size, context_length)
        logits = model(x_batch)
        loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / 100
    print(f"✅ Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

# Save model weights
torch.save(model.state_dict(), "model/gpt_nano.pth")
print("✅ Model training complete. Weights saved to model/gpt_nano.pth")
