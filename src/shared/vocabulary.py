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
    text = text.lower() # passa a minúscules perquè "Dog" i "dog" siguin la mateixa paraula
    tokens = re.findall(r"[a-z0-9']+", text) # separa el text en paraules però manté els apostrofs:
    return tokens        # "A dog, running! It's happy." --> ["a", "dog", "running", "it's", "happy"]


class Vocabulary:
    def __init__(self):
        self.word2idx: dict[str, int] = {} # diccionari paraula → índex: {"dog": 4, "cat": 5, ...}
        self.idx2word: dict[int, str] = {} # diccionari invers índex → paraula: {4: "dog", 5: "cat", ...}
        self.idx = 0 # comptador d'índex. Comença a 0 i s'incrementa cada vegada que s'afegeix una paraula
        for tok in (PAD, START, END, UNK):
            self.add_word(tok) # afegeix els 4 tokens especials: {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}

    def add_word(self, word: str) -> None:
        if word not in self.word2idx: # comprova que la paraula no estigui ja al vocabulari per evitar duplicats
            self.word2idx[word] = self.idx # afegeix la paraula al diccionari paraula → índex
            self.idx2word[self.idx] = word # afegeix la paraula al diccionari invers índex → paraula
            self.idx += 1 # incrementa el comptador per assignar un índex diferent a la següent paraula

    def __call__(self, word: str) -> int: # permet fer vocab("dog") en comptes de vocab.word2idx["dog"]
        return self.word2idx.get(word, self.word2idx[UNK]) # si la paraula no hi és retorna l'índex de <unk> (3)

    def __len__(self) -> int:
        return len(self.word2idx) # retorna el nombre total de paraules al vocabulari

    def encode(self, caption: str, add_special: bool = True) -> list[int]:
        tokens = simple_tokenize(caption) # tokenitza la frase: "a dog runs" → ["a", "dog", "runs"]
        ids = [self(t) for t in tokens] # converteix cada token en el seu índex. Si no hi és → 3 (<unk>)
        if add_special:
            ids = [self(START)] + ids + [self(END)] # afegeix <start> (1) al principi i <end> (2) al final
        return ids # retorna llista d'enters: [1, 4, 27, 83, 2]

    def decode(self, ids: list[int], skip_special: bool = True) -> str: # fa el procés invers: [1, 4, 5, 2] → "a dog cat"
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), UNK) # per cada índex troba la paraula corresponent (si no hi és, <unk>)
            if skip_special and w in (PAD, START): # si troba <pad> o <start>, els salta (no els afegeix)
                continue
            if skip_special and w == END: # si troba <end>, para de decodificar
                break
            words.append(w) # afegeix la paraula a la llista
        return " ".join(words) # uneix totes les paraules amb espais: ["a", "dog"] → "a dog"


def load_glove_weights(glove_path: str | Path, vocab: Vocabulary) -> tuple["torch.Tensor", int]:
    """Carrega vectors GloVe i construeix una matriu de pesos per al vocabulari.

    Les paraules del vocabulari que no apareixen a GloVe s'inicialitzen aleatòriament.
    Retorna (weight_matrix [vocab_size, glove_dim], glove_dim).

    Descàrrega GloVe: https://nlp.stanford.edu/data/glove.6B.zip
    Recomanat: glove.6B.300d.txt
    """
    import torch

    print(f"[glove] carregant {glove_path}...")
    glove: dict[str, list[float]] = {} # diccionari buit on guardarem paraula → vector
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split() # cada línia és: "dog 0.12 -0.34 0.56 ..."
            glove[parts[0]] = [float(x) for x in parts[1:]] # guarda la paraula i el seu vector de floats

    glove_dim = len(next(iter(glove.values())))  # detecta la dimensió automàticament (50, 100, 200 o 300)
    found, total = 0, len(vocab) # comptadors per saber quantes paraules del vocab trobem a GloVe

    weights = torch.randn(total, glove_dim) * 0.01 # matriu inicialitzada aleatòriament (molt petit) per a paraules fora de GloVe
    weights[0] = 0  # <pad> → vector zero perquè el padding no contribueixi a res

    for word, idx in vocab.word2idx.items():
        if word in glove:
            weights[idx] = torch.tensor(glove[word]) # substitueix el vector aleatori pel vector GloVe real
            found += 1 # incrementa el comptador de paraules trobades

    print(f"[glove] {found}/{total} paraules del vocabulari trobades a GloVe ({glove_dim}d)")
    return weights, glove_dim # retorna la matriu de pesos [vocab_size, glove_dim] i la dimensió


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
        binary = w2v_path.suffix == ".bin" # detecta automàticament si és binari per l'extensió del fitxer

    print(f"[word2vec] carregant {w2v_path} (binary={binary})...")
    wv = KeyedVectors.load_word2vec_format(str(w2v_path), binary=binary) # carrega els vectors amb gensim

    w2v_dim = wv.vector_size # dimensió dels vectors (normalment 300)
    found, total = 0, len(vocab) # comptadors per saber quantes paraules del vocab trobem a Word2Vec

    weights = torch.randn(total, w2v_dim) * 0.01 # matriu inicialitzada aleatòriament per a paraules fora de Word2Vec
    weights[0] = 0  # <pad> → vector zero perquè el padding no contribueixi a res

    for word, idx in vocab.word2idx.items():
        if word in wv:
            weights[idx] = torch.tensor(wv[word]) # substitueix el vector aleatori pel vector Word2Vec real
            found += 1 # incrementa el comptador de paraules trobades

    print(f"[word2vec] {found}/{total} paraules del vocabulari trobades a Word2Vec ({w2v_dim}d)")
    return weights, w2v_dim # retorna la matriu de pesos [vocab_size, w2v_dim] i la dimensió


def build_vocab(captions_csv: str | Path, threshold: int = 5) -> Vocabulary:
    df = pd.read_csv(captions_csv) # llegeix el CSV de captions com a DataFrame
    counter: Counter[str] = Counter() # crea un counter buit per comptar freqüències de paraules
    for cap in df["caption"].astype(str):
        counter.update(simple_tokenize(cap)) # tokenitza cada caption i actualitza el counter amb les seves paraules

    vocab = Vocabulary() # crea un vocabulari buit amb els 4 tokens especials ja afegits
    for word, count in counter.items(): # itera sobre totes les paraules i les seves freqüències
        if count >= threshold: # si la paraula apareix almenys `threshold` vegades, l'afegeix al vocabulari
            vocab.add_word(word) # paraules rares (< threshold) s'ignoraran i quedaran com <unk>
    return vocab


def build_vocab_hf(hf_dataset, threshold: int = 5) -> Vocabulary:
    """Construeix un Vocabulary a partir del dataset HuggingFace nlphuji/flickr30k.

    Args:
        hf_dataset: resultat de load_dataset('nlphuji/flickr30k', ...) → clau 'test'
        threshold:  mínim d'aparicions per incloure una paraula al vocabulari
    """
    counter: Counter[str] = Counter()
    for row in hf_dataset["test"]:
        for cap in row["caption"]:           # cada fila té una llista de 5 captions
            counter.update(simple_tokenize(cap))

    vocab = Vocabulary()
    for word, count in counter.items():
        if count >= threshold:
            vocab.add_word(word)
    return vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", default="data/flickr8k/captions.txt") # ruta al csv de captions
    p.add_argument("--out", default="data/flickr8k/vocab.pkl") # ruta on guardar el vocab entrenat en un .pkl
    p.add_argument("--threshold", type=int, default=5) # mínim de vegades que ha d'aparèixer una paraula per entrar al vocab
    args = p.parse_args()

    vocab = build_vocab(args.captions, args.threshold) # construeix vocabulari a partir del csv i threshold
    Path(args.out).parent.mkdir(parents=True, exist_ok=True) # crea carpeta on guardar el vocab si no existeix
    with open(args.out, "wb") as f: # obre en mode escriptura binària
        pickle.dump(vocab, f) # guarda el vocabulari serialitzat al fitxer .pkl
    print(f"Vocab size: {len(vocab)} (threshold={args.threshold})") # imprimeix mida del vocabulari i threshold utilitzat
    print(f"Saved to {args.out}") # imprimeix ruta on s'ha guardat el vocabulari


if __name__ == "__main__": # si fas import vocabulary.py des d'un altre fitxer, no s'executa main(),
    main()                 # però si executes `python vocabulary.py` sí que s'executa
