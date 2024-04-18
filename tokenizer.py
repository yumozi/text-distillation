# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import struct
import argparse
from typing import List
from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity as cosine_similarity2
import numpy as np
from collections import Counter

TOKENIZER_MODEL = "tokenizer.model" # the llama sentencepiece tokenizer model

class Tokenizer:
    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        #print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def export(self):

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):

            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8') # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)


def vectorize(tokens: List[int], vocab_size: int) -> np.array:
    """Convert tokenized text to a vector."""
    vector = np.zeros(vocab_size)
    token_counts = Counter(tokens)
    for token, count in token_counts.items():
        vector[token] = count
    return vector


def cosine_similarity(vec1: np.array, vec2: np.array) -> float:
    """Calculate cosine similarity between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def calculate_text_similarity(text1: str, text2: str, tokenizer_model: str = None) -> float:
    """
    Calculate the cosine similarity between two texts using a specified tokenizer model.

    Args:
    text1 (str): First text to compare.
    text2 (str): Second text to compare.
    tokenizer_model (str, optional): Path to a custom tokenizer model file. Defaults to None.

    Returns:
    float: Cosine similarity between the vector representations of text1 and text2.
    """
    tokenizer = Tokenizer(tokenizer_model)

    tokens1 = tokenizer.encode(text1, bos=True, eos=False)
    tokens2 = tokenizer.encode(text2, bos=True, eos=False)

    vocab_size = tokenizer.n_words
    vec1 = vectorize(tokens1, vocab_size)
    vec2 = vectorize(tokens2, vocab_size)

    similarity = cosine_similarity(vec1, vec2)
    return similarity


# Load model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def calculate_text_similarity2(text1, text2):
    # Tokenize text
    inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True)

    # Generate embeddings
    with torch.no_grad():
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

    # Calculate cosine similarity
    similarity = cosine_similarity2(embeddings1, embeddings2)

    return similarity.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer-model", type=str, help="optional path to custom tokenizer ")
    args = parser.parse_args()

    t = Tokenizer(args.tokenizer_model)
    t.export()

    # e.g. cosine similarity usage
    # text1 = "The weather is sunny."
    # text2 = "It's a bright sunny day."
    # similarity_score = calculate_text_similarity2(text1, text2)
    # print("Similarity Score:", similarity_score)