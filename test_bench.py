from model.tokenizer import CharTokenizer
import re
import unicodedata

# 1. Load text
print("📂 Loading TinyStories dataset...")
with open("data/tinystories_train.txt", "r", encoding="utf-8") as f:
    train_text = f.read()
with open("data/tinystories_valid.txt", "r", encoding="utf-8") as f:
    val_text = f.read()

# 2. Build tokenizer from full dataset (train + val)
print("🧠 Building tokenizer...")

def clean_text(s):
    # Lowercase
    s = s.lower()
    # Normalize Unicode (e.g., é → e)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    # Replace fancy quotes/dashes/ellipsis
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("—", "-").replace("–", "-").replace("…", "...")
    # Keep only allowed chars
    s = re.sub(r"[^a-z0-9.,!?;:'\"()\n -]", "", s)
    return s

train_text = clean_text(train_text)
val_text = clean_text(val_text)
full_text = train_text + val_text
tokenizer = CharTokenizer(full_text)

text_sample = "abc xyz! hello\n"
ids = tokenizer.encode(text_sample)
assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
assert tokenizer.decode(ids) == text_sample
print("Vocab size:", tokenizer.vocab_size)
print(ids)
