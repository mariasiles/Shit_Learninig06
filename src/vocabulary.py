"""Vocabulary builder for Flickr8k captions.

Reads a captions CSV (image,caption) and builds a word<->index mapping
keeping only words that appear at least `threshold` times.
"""
from __future__ import annotations

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Special tokens
PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"


def simple_tokenize(text: str) -> list[str]:
    """Lowercase and keep alphanumeric tokens. No NLTK dependency."""
    text = text.lower()
    # split on non-word characters but keep apostrophes inside words
    tokens = re.findall(r"[a-z0-9']+", text)
    return tokens


class Vocabulary:
    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.idx = 0
        for tok in (PAD, START, END, UNK):
            self.add_word(tok)

    def add_word(self, word: str) -> None:
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx[UNK])

    def __len__(self) -> int:
        return len(self.word2idx)

    def encode(self, caption: str, add_special: bool = True) -> list[int]:
        tokens = simple_tokenize(caption)
        ids = [self(t) for t in tokens]
        if add_special:
            ids = [self(START)] + ids + [self(END)]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), UNK)
            if skip_special and w in (PAD, START):
                continue
            if skip_special and w == END:
                break
            words.append(w)
        return " ".join(words)


def build_vocab(captions_csv: str | Path, threshold: int = 5) -> Vocabulary:
    df = pd.read_csv(captions_csv)
    counter: Counter[str] = Counter()
    for cap in df["caption"].astype(str):
        counter.update(simple_tokenize(cap))

    vocab = Vocabulary()
    for word, count in counter.items():
        if count >= threshold:
            vocab.add_word(word)
    return vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", default="data/flickr8k/captions.txt")
    p.add_argument("--out", default="data/flickr8k/vocab.pkl")
    p.add_argument("--threshold", type=int, default=5)
    args = p.parse_args()

    vocab = build_vocab(args.captions, args.threshold)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocab size: {len(vocab)} (threshold={args.threshold})")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
