
if __name__ == "__main__":
    sample_text = "batcgus"
    set_1 = set(sample_text)
    list_1 = list(set_1)
    list_1_sorted = sorted(list_1)
    char = enumerate(list_1_sorted)
    print("Unique characters in sample text:", set_1)
    print("List of unique characters:", list_1)
    print("Sorted list of unique characters:", list_1_sorted)
    print("Enumerated characters:", list(char))