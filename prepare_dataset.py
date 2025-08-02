# prepare_dataset.py

import os
import requests
import json

def download_tinystories():
    url = "https://raw.githubusercontent.com/roneneldan/TinyStories/main/TinyStoriesV2-GPT4/train.json"
    raw_path = "data/tinystories.json"
    text_path = "data/tinystories.txt"

    # Step 1: download JSON if not exists
    if not os.path.exists(raw_path):
        print("Downloading TinyStories JSON...")
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception(f"Failed to download dataset: {r.status_code}")
        with open(raw_path, 'wb') as f:
            f.write(r.content)
        print("Download complete.")

    # Step 2: extract and save as .txt
    if not os.path.exists(text_path):
        print("Extracting text from JSON...")
        with open(raw_path, 'r', encoding='utf-8') as f:
            stories = json.load(f)

        with open(text_path, 'w', encoding='utf-8') as f:
            for story in stories:
                f.write(story['story'] + "\n\n")
        print("Text file ready.")

    else:
        print("TinyStories already prepared.")

if __name__ == "__main__":
    download_tinystories()
