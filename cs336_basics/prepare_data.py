import numpy as np
from .tokenizer import Tokenizer

tokenizer = Tokenizer.from_files("owt_32k_vocab.txt", "owt_32k_merges.txt", special_tokens=["<|endoftext|>"])

'''
with open("data/owt_valid.txt", "r", encoding="utf-8") as f:
    tokens = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
'''

tokens = np.load("owt_valid_tokens.npy")
print(tokens.shape)
print(tokens.dtype)
print(tokens.min())
print(tokens.max())




