# generate.py

import torch
from model.transformer import TransformerModel
import pickle

# Load tokenizer
with open("data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Model parameters (must match what was used during training)
vocab_size = tokenizer.vocab_size
embed_dim = 128
num_heads = 4
num_layers = 2
context_length = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and weights
model = TransformerModel(vocab_size, embed_dim, num_heads, num_layers, context_length).to(device)
model.load_state_dict(torch.load("model/gpt_nano.pth", map_location=device))
model.eval()

# Text generation function
@torch.no_grad()
def generate_text(prompt, max_new_tokens=100):
    # Encode prompt to token IDs
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # shape: (1, seq_len)

    for _ in range(max_new_tokens):
        input_crop = input_tensor[:, -context_length:]  # Crop to context window
        logits = model(input_crop)
        temperature = 1.1  # You can try 0.8 or 1.2 to adjust creativity
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat((input_tensor, next_token), dim=1)

    output_ids = input_tensor.squeeze().tolist()
    print("🔢 Output token IDs:", output_ids)
    return tokenizer.decode(output_ids)

# Example usage
if __name__ == "__main__":
    prompt = input("📝 Enter prompt: ")
    generated = generate_text(prompt, max_new_tokens=200)
    print("\n🧠 Generated Text:\n")
    print(generated)
