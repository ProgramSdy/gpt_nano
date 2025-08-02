# train.py

from model.tokenizer import CharTokenizer

if __name__ == "__main__":
    sample_text = "hello nano gpt!"
    tokenizer = CharTokenizer(sample_text)

    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)

    print("Original text:", "hello")
    print("Encoded:", encoded)
    print("Decoded:", decoded)
    print("Vocab size:", tokenizer.vocab_size)
