# train.py

import os
import numpy as np
import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler  # ✅ new AMP API
from tqdm import tqdm
import pickle
from model.transformer import TransformerModel

# ----------------
# Config
# ----------------
DATA_TRAIN_PATH = "data/train.npy"
DATA_VAL_PATH   = "data/val.npy"
TOKENIZER_PATH  = "data/tokenizer.pkl"
CKPT_DIR        = "model"
CKPT_BEST       = os.path.join(CKPT_DIR, "gpt_nano_best.pth")
CKPT_LAST       = os.path.join(CKPT_DIR, "gpt_nano.pth")

# Model / training hyperparams
embed_dim       = 256
num_heads       = 8
num_layers      = 6
context_length  = 256
batch_size      = 128
learning_rate   = 3e-4
num_epochs      = 20

# Keep epochs predictable
MAX_STEPS_PER_EPOCH = 1500
VAL_STEPS            = 20

# Quick qualitative sample each epoch
PRINT_SAMPLE_EVERY_EPOCH = True
SAMPLE_PROMPT            = "Once upon a time"
SAMPLE_MAX_NEW_TOKENS    = 200
SAMPLE_TEMPERATURE       = 0.9
SAMPLE_TOP_K             = 50

# ----------------
# Setup
# ----------------
os.makedirs(CKPT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")  # good on Ampere+

# Load memmapped arrays (uint16) and wrap as torch tensors
train_mm = np.memmap(DATA_TRAIN_PATH, dtype=np.uint16, mode='r')
val_mm   = np.memmap(DATA_VAL_PATH,   dtype=np.uint16, mode='r')
train_data = torch.from_numpy(train_mm).long()
val_data   = torch.from_numpy(val_mm).long()

# Load tokenizer to get vocab size and decode/encode
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
vocab_size = tokenizer.vocab_size

# Model / opt / loss
model = TransformerModel(vocab_size, embed_dim, num_heads, num_layers, context_length).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler('cuda') if device.type == 'cuda' else GradScaler(enabled=False)

# ----------------
# Data helpers
# ----------------
def get_batch(data: torch.Tensor, batch: int, ctx: int):
    ix = torch.randint(len(data) - ctx - 1, (batch,))
    x = torch.stack([data[i:i+ctx] for i in ix])
    y = torch.stack([data[i+1:i+ctx+1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

@torch.no_grad()
def evaluate(model, steps=VAL_STEPS):
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = get_batch(val_data, batch_size, context_length)
    with autocast('cuda', enabled=(device.type == 'cuda')):
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))

# ----------------
# Sampling helpers (temperature + top-k) for quick qualitative checks
# ----------------
@torch.no_grad()
def sample_next(logits_last_t, temperature=1.0, top_k=None):
    # logits_last_t: (B, vocab_size)
    logits = logits_last_t / max(1e-8, temperature)
    if top_k is not None and 0 < top_k < logits.size(-1):
        v, idx = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(v, dim=-1)
        next_tok = idx.gather(1, torch.multinomial(probs, num_samples=1))
    else:
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
    return next_tok

@torch.no_grad()
def quick_generate(prompt, max_new_tokens=200, temperature=0.9, top_k=50):
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]  # (1, T)
    for _ in range(max_new_tokens):
        x_cond = x[:, -context_length:]  # crop to context
        with autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(x_cond)
        next_tok = sample_next(logits[:, -1, :], temperature=temperature, top_k=top_k)
        x = torch.cat([x, next_tok], dim=1)
    return tokenizer.decode(x[0].tolist())

# ----------------
# Train
# ----------------
best_val = float("inf")

for epoch in range(1, num_epochs + 1):
    model.train()
    est_steps = max(1, (len(train_data) // batch_size) // 10)
    steps_per_epoch = min(est_steps, MAX_STEPS_PER_EPOCH)

    print(f"\n🚀 Epoch {epoch}/{num_epochs} | steps_per_epoch={steps_per_epoch}")
    running = 0.0

    for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}", leave=False):
        x, y = get_batch(train_data, batch_size, context_length)

        with autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()

    avg_loss = running / steps_per_epoch
    val_loss = evaluate(model)
    print(f"✅ Epoch {epoch} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), CKPT_BEST)
        print(f"💾 Saved best checkpoint → {CKPT_BEST} (val_loss={best_val:.4f})")

    if PRINT_SAMPLE_EVERY_EPOCH:
        sample = quick_generate(
            SAMPLE_PROMPT,
            max_new_tokens=SAMPLE_MAX_NEW_TOKENS,
            temperature=SAMPLE_TEMPERATURE,
            top_k=SAMPLE_TOP_K,
        )
        print("📝 Sample:")
        print(sample)
        print("-" * 80)

# always save last
torch.save(model.state_dict(), CKPT_LAST)
print(f"✅ Training complete. Last weights saved → {CKPT_LAST}")
print(f"🏆 Best val_loss: {best_val:.4f} (checkpoint at {CKPT_BEST})")
