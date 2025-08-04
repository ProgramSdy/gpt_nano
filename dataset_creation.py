# dataset_creation.py

import numpy as np
import os
import pickle
from model.tokenizer import CharTokenizer

# 1. Load text
print("📂 Loading TinyStories dataset...")
with open("data/tinystories_train.txt", "r", encoding="utf-8") as f:
    train_text = f.read()
with open("data/tinystories_valid.txt", "r", encoding="utf-8") as f:
    val_text = f.read()

# 2. Build tokenizer from full dataset (train + val)
print("🧠 Building tokenizer...")
full_text = train_text + val_text
tokenizer = CharTokenizer(full_text)
print(f"🔡 Vocab size: {tokenizer.vocab_size}")

# 3. Function to count total tokens
def count_tokens(text, tokenizer, chunk_size=100_000):
    total = 0
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        total += len(tokenizer.encode(chunk))
    return total

# 4. Memory-efficient encode and save to .npy using memmap
def write_tokens_to_memmap(text, tokenizer, path, total_tokens, chunk_size=100_000):
    memmap_array = np.memmap(path, dtype=np.uint16, mode='w+', shape=(total_tokens,))
    idx = 0
    total_chunks = (len(text) + chunk_size - 1) // chunk_size

    for chunk_idx, i in enumerate(range(0, len(text), chunk_size)):
        chunk = text[i:i + chunk_size]
        encoded = tokenizer.encode(chunk)
        memmap_array[idx:idx+len(encoded)] = encoded
        idx += len(encoded)
        print(f"  📦 Encoded chunk {chunk_idx + 1} / {total_chunks} | tokens: {len(encoded)}")

    memmap_array.flush()
    print(f"✅ Saved to {path}")

# 5. Process train
print("🔄 Encoding train set...")
train_total_tokens = count_tokens(train_text, tokenizer)
write_tokens_to_memmap(train_text, tokenizer, "data/train.npy", train_total_tokens)

# 6. Process val
print("🔄 Encoding validation set...")
val_total_tokens = count_tokens(val_text, tokenizer)
write_tokens_to_memmap(val_text, tokenizer, "data/val.npy", val_total_tokens)

# 7. Save tokenizer
with open("data/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("💾 Tokenizer saved to data/tokenizer.pkl")
