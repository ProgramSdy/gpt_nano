

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