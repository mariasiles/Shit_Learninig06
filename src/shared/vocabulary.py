from __future__ import annotations # per escriure anotacions modernes com: list[str]

import argparse # per executar fitxers des del terminal amb arguments (--captions, --out, --threshold)
import pickle # per guardar el vocabulari entrenat en un .pkl
import re # per a la funció simple_tokenize, que utilitza re per a tokenitzar el text
from collections import Counter # counter de freqs: Counter(["dog","cat","dog"]) --> Counter({"dog":2,"cat":1})
from pathlib import Path # per rutes de fitxers de manera neta 

import pandas as pd # per llegir el CSV de captions i processar-lo com a DataFrame

# Special tokens
PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"


def simple_tokenize(text: str) -> list[str]:
    text = text.lower() # passa a minúscules
    tokens = re.findall(r"[a-z0-9']+", text) # separa però no per apostrofs: 
    return tokens        # "A dog, running! It's happy." --> ["a", "dog", "running", "it's", "happy"]


class Vocabulary:
    def __init__(self):
        self.word2idx: dict[str, int] = {} # {"dog": 4, "cat": 5, ...}
        self.idx2word: dict[int, str] = {} # {4: "dog", 5: "cat", ...}
        self.idx = 0 # comptador d'índex. Comença a 0.
        for tok in (PAD, START, END, UNK):
            self.add_word(tok) # {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}

    def add_word(self, word: str) -> None:
        if word not in self.word2idx: # si paraula no està al vocabulari, l'afegeix
            self.word2idx[word] = self.idx # als dos diccionaris
            self.idx2word[self.idx] = word
            self.idx += 1 # incrementa comptador per la següent paraula

    def __call__(self, word: str) -> int: # per fer vocab(word) en comptes de vocab.word2idx[word]
        return self.word2idx.get(word, self.word2idx[UNK]) # si word no hi és retorna l'índex de UNK (3)

    def __len__(self) -> int:
        return len(self.word2idx)

    def encode(self, caption: str, add_special: bool = True) -> list[int]:
        tokens = simple_tokenize(caption) # tokenitza la frase amb la funció simple_tokenize
        ids = [self(t) for t in tokens] # converteix cada token en el seu index. Si no hi és --> self(t) --> 3 (UNK)
        if add_special: 
            ids = [self(START)] + ids + [self(END)] # afegeix token inicial i final
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str: # fa procés invers [1, 4, 5, 2] --> "a dog cat"
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), UNK) # per cada index troba la paraula (si no hi és, UNK)
            if skip_special and w in (PAD, START): # si troba pad o start, no afegeix la paraula
                continue
            if skip_special and w == END: # si troba end, acaba amb la frase
                break
            words.append(w) # afegeix la paraula a la llista de paraules
        return " ".join(words) # les uneix amb espais


def load_glove_weights(glove_path: str | Path, vocab: Vocabulary) -> tuple["torch.Tensor", int]:
    """Carrega vectors GloVe i construeix una matriu de pesos per al vocabulari.

    Les paraules del vocabulari que no apareixen a GloVe s'inicialitzen aleatòriament.
    Retorna (weight_matrix [vocab_size, glove_dim], glove_dim).

    Descàrrega GloVe: https://nlp.stanford.edu/data/glove.6B.zip
    Recomanat: glove.6B.300d.txt
    """
    import torch

    print(f"[glove] carregant {glove_path}...")
    glove: dict[str, list[float]] = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            glove[parts[0]] = [float(x) for x in parts[1:]]

    glove_dim = len(next(iter(glove.values())))  # detecta la dimensió automàticament (50, 100, 200 o 300)
    found, total = 0, len(vocab)

    # matriu de pesos inicialitzada aleatòriament per a les paraules fora de GloVe
    weights = torch.randn(total, glove_dim) * 0.01
    weights[0] = 0  # <pad> → vector zero

    for word, idx in vocab.word2idx.items():
        if word in glove:
            weights[idx] = torch.tensor(glove[word])
            found += 1

    print(f"[glove] {found}/{total} paraules del vocabulari trobades a GloVe ({glove_dim}d)")
    return weights, glove_dim


def load_word2vec_weights(w2v_path: str | Path, vocab: Vocabulary, binary: bool | None = None) -> tuple["torch.Tensor", int]:
    """Carrega vectors Word2Vec i construeix una matriu de pesos per al vocabulari.

    Les paraules del vocabulari que no apareixen a Word2Vec s'inicialitzen aleatòriament.
    Retorna (weight_matrix [vocab_size, w2v_dim], w2v_dim).

    Requereix: pip install gensim
    Formats suportats:
      - Binari (.bin): GoogleNews-vectors-negative300.bin
      - Text (.txt):   capçalera "vocab_size dim" + una paraula per línia
    """
    import torch
    from gensim.models import KeyedVectors

    w2v_path = Path(w2v_path)
    if binary is None:
        binary = w2v_path.suffix == ".bin"

    print(f"[word2vec] carregant {w2v_path} (binary={binary})...")
    wv = KeyedVectors.load_word2vec_format(str(w2v_path), binary=binary)

    w2v_dim = wv.vector_size
    found, total = 0, len(vocab)

    weights = torch.randn(total, w2v_dim) * 0.01
    weights[0] = 0  # <pad> → vector zero

    for word, idx in vocab.word2idx.items():
        if word in wv:
            weights[idx] = torch.tensor(wv[word])
            found += 1

    print(f"[word2vec] {found}/{total} paraules del vocabulari trobades a Word2Vec ({w2v_dim}d)")
    return weights, w2v_dim


def build_vocab(captions_csv: str | Path, threshold: int = 5) -> Vocabulary:
    df = pd.read_csv(captions_csv)
    counter: Counter[str] = Counter() # crea un counter buit
    for cap in df["caption"].astype(str):
        counter.update(simple_tokenize(cap)) # tokenitza cada caption i actualitza el counter

    vocab = Vocabulary() # crea un vocabulari buit amb els tokens especials 
    for word, count in counter.items(): # itera sobre les paraules i les seves freqüències al counter
        if count >= threshold: # si com a minim apareix `threshold` vegades, afegeix la paraula al vocabulari
            vocab.add_word(word)
    return vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", default="data/flickr8k/captions.txt") # ruta al csv de captions
    p.add_argument("--out", default="data/flickr8k/vocab.pkl") # ruta on guardar el vocab entrenat en un .pkl
    p.add_argument("--threshold", type=int, default=5) # per definir un altre threshold que no sigui 5
    args = p.parse_args() 

    vocab = build_vocab(args.captions, args.threshold) # construeix vocabulari a partir del csv i threshold
    Path(args.out).parent.mkdir(parents=True, exist_ok=True) # crea carpeta on guardar el vocab si no existeix
    with open(args.out, "wb") as f: # obre en mode escriptura binària
        pickle.dump(vocab, f) # guarda vocab al fitxer
    print(f"Vocab size: {len(vocab)} (threshold={args.threshold})") # imprimeix mida del vocabulari i threshold utilitzat
    print(f"Saved to {args.out}") # imprimeix ruta on s'ha guardat el vocabulari


if __name__ == "__main__": # si fas import vocabulary.py des d'un altre fitxer, no s'executa main(), 
    main()                 # però si executes `python vocabulary.py` sí que s'executa
