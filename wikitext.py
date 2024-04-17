import os
import json
from typing import List
from functools import partial
import numpy as np

import torch
import torch.distributed as dist
from tqdm import tqdm
from datasets import load_dataset
from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"

class Task:

    @staticmethod
    def download():
        """Downloads and saves the Wikitext-2 dataset to a specified directory."""
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        for split in ['train', 'validation', 'test']:
            with open(os.path.join(DATA_CACHE_DIR, f"wikitext-2_{split}.json"), 'w') as f:
                json.dump(dataset[split]['text'], f)
            print(f"Wikitext-2 {split} data saved.")

    @staticmethod
    def encode_data(vocab_size):
        """Encodes the downloaded text data using a custom Tokenizer."""
        assert vocab_size > 0, "Vocab size must be positive"
        tokenizer_model = None
        tokenizer = Tokenizer(tokenizer_model)

        for split in ['train', 'validation', 'test']:
            input_file = os.path.join(DATA_CACHE_DIR, f"wikitext-2_{split}.json")
            output_file = input_file.replace('.json', '.bin')

            with open(input_file, 'r') as f:
                texts = json.load(f)

            encoded_texts = []
            for text in tqdm(texts, desc=f"Encoding {split} data"):
                tokens = tokenizer.encode(text, bos=True, eos=True)
                encoded_texts.extend(tokens)

            # Save the encoded data to a binary file
            with open(output_file, 'wb') as f:
                torch.tensor(encoded_texts, dtype=torch.int16).numpy().tofile(f)
            print(f"Encoded {split} data saved to {output_file}.")

    @staticmethod
    def iter_batches(batch_size, max_seq_len, vocab_size, split='train', device='cpu'):
        """Yields batches of data suitable for model training."""
        file_path = os.path.join(DATA_CACHE_DIR, f"wikitext-2_{split}.bin")
        data = np.fromfile(file_path, dtype=np.int16)

        # Ensure we can make full batches
        num_batches = len(data) // (batch_size * max_seq_len)
        data = data[:num_batches * batch_size * max_seq_len]
        data = data.reshape(batch_size, num_batches * max_seq_len)

        for i in range(num_batches):
            start = i * max_seq_len
            end = start + max_seq_len + 1
            batch_data = torch.from_numpy(data[:, start:end].astype(np.int64)).to(device)
            x = batch_data[:, :-1]
            y = batch_data[:, 1:]
            yield x, y

# -----------------------------------------------------------------------------
# CLI for processing the dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "encode_data", "iter_batches"])
    parser.add_argument("--vocab_size", type=int, default=2048, help="Vocabulary size for tokenizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading.")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length for batches.")
    parser.add_argument("--split", type=str, default='train', choices=['train', 'validation', 'test'], help="Dataset split to process.")
    parser.add_argument("--device", type=str, default='cpu', help="Device for tensors (e.g., 'cpu' or 'cuda').")
    args = parser.parse_args()

    if args.stage == "download":
        Task.download()
    elif args.stage == "encode_data":
        Task.encode_data(vocab_size=args.vocab_size)
    elif args.stage == "iter_batches":
        for x, y in Task.iter_batches(args.batch_size, args.max_seq_len, args.vocab_size, args.split, args.device):
            print("Batch X:", x)
            print("Batch Y:", y)
            break