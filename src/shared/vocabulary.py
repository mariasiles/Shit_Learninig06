from __future__ import annotations # per escriure anotacions modernes com: list[str]

import argparse # per executar fitxers des del terminal amb arguments (--captions, --out, --threshold)
import pickle # per guardar el vocabulari entrenat en un .pkl
import re # per a la funció simple_tokenize, que utilitza re per a tokenitzar el text
from collections import Counter # counter de freqs: Counter(["dog","cat","dog"]) --> Counter({"dog":2,"cat":1})
from pathlib import Path # per rutes de fitxers de manera neta 

import numpy as np # per crear la matriu d'embeddings de GloVe
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

        # Matriu d'embeddings GloVe (opcional). Omple's amb build_vocab_glove().
        # Si és None, el model usarà embeddings aleatoris entrenables (mode estàndard).
        self.glove_embeddings: np.ndarray | None = None

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


# =============================================================================
# OPCIÓ 1 — Vocabulari estàndard (sense semàntica)
# Cada paraula rep un índex numèric basat en freqüència d'aparició.
# "gat" i "felí" seran dos nombres sense cap relació entre ells.
# =============================================================================

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


# =============================================================================
# OPCIÓ 2 — Vocabulari amb GloVe (amb semàntica)
# Cada paraula rep un vector de 50/100/200/300 dimensions preentrenat amb GloVe.
# "gat" i "felí" tindran vectors molt similars perquè apareixen en contextos similars.
#
# Com funciona GloVe?
#   - Entrenat sobre milers de milions de paraules de text
#   - Paraules que apareixen en contextos similars → vectors similars
#   - Similitud cosinus entre vectors ≈ similitud semàntica
#
# El fitxer glove_path ha de ser el .txt descarregat de:
#   https://nlp.stanford.edu/projects/glove/  (glove.6B.zip → glove.6B.300d.txt)
# =============================================================================

def build_vocab_glove(captions_csv: str | Path,
                      glove_path: str | Path,
                      threshold: int = 5,
                      glove_dim: int = 300) -> Vocabulary:
    """
    Construeix el vocabulari igual que build_vocab(), però a més carrega
    els vectors GloVe per a cada paraula i els guarda a vocab.glove_embeddings.

    Retorna un Vocabulary amb:
      - word2idx / idx2word  (igual que l'opció estàndard)
      - glove_embeddings: np.ndarray de forma (vocab_size, glove_dim)
        que el model pot usar per inicialitzar la capa nn.Embedding
    """

    # --- Pas 1: construïm el vocabulari de la mateixa manera que build_vocab ---
    df = pd.read_csv(captions_csv)
    counter: Counter[str] = Counter()
    for cap in df["caption"].astype(str):
        counter.update(simple_tokenize(cap)) # comptador de freqüències de paraules

    vocab = Vocabulary() # inicia amb els 4 tokens especials (<pad>, <start>, <end>, <unk>)
    for word, count in counter.items():
        if count >= threshold: # només afegeix paraules que apareixen prou vegades
            vocab.add_word(word)

    # --- Pas 2: llegim el fitxer GloVe i guardem els vectors en un diccionari ---
    print(f"Carregant GloVe des de {glove_path} ...")
    glove_vectors: dict[str, np.ndarray] = {} # {"dog": array([0.1, -0.3, ...]), ...}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split() # cada línia: "dog 0.1 -0.3 0.7 ..."
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32) # converteix els números a float
            glove_vectors[word] = vector # guarda el vector per a la paraula

    # --- Pas 3: construïm la matriu d'embeddings alineada amb el nostre vocabulari ---
    # Mida: (nombre de paraules al vocab, dimensió GloVe)
    # Inicialitzem amb zeros. Les paraules sense GloVe quedaran a zero (el model les aprendrà).
    embedding_matrix = np.zeros((len(vocab), glove_dim), dtype=np.float32)

    n_found = 0 # comptador de paraules trobades a GloVe
    for word, idx in vocab.word2idx.items():
        if word in glove_vectors: # si la paraula té vector GloVe, l'inserim a la matriu
            embedding_matrix[idx] = glove_vectors[word]
            n_found += 1
        # si no hi és (paraula rara o token especial), deixem el vector a zeros

    coverage = n_found / len(vocab) * 100 # % de paraules del vocab cobertes per GloVe
    print(f"GloVe cobreix {n_found}/{len(vocab)} paraules del vocab ({coverage:.1f}%)")

    # Guardem la matriu dins el vocabulari perquè el model la pugui usar directament
    vocab.glove_embeddings = embedding_matrix # shape: (vocab_size, glove_dim)

    return vocab


# =============================================================================
# COM USAR ELS EMBEDDINGS GLOVE AL MODEL (exemple per al fitxer model.py)
# =============================================================================
#
#   vocab = build_vocab_glove("captions.csv", "glove.6B.300d.txt", glove_dim=300)
#
#   import torch
#   glove_tensor = torch.tensor(vocab.glove_embeddings)  # (vocab_size, 300)
#
#   self.embedding = nn.Embedding(len(vocab), embed_dim)
#   self.embedding.weight = nn.Parameter(glove_tensor)   # inicialitza amb GloVe
#   self.embedding.weight.requires_grad = True            # True = fine-tuning (recomanat)
#                                                         # False = vectors fixos
#
# =============================================================================


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", default="data/flickr8k/captions.txt") # ruta al csv de captions
    p.add_argument("--out", default="data/flickr8k/vocab.pkl") # ruta on guardar el vocab entrenat en un .pkl
    p.add_argument("--threshold", type=int, default=5) # per definir un altre threshold que no sigui 5
    # --glove: si s'especifica, usa GloVe. Si no, usa l'opció estàndard sense semàntica.
    p.add_argument("--glove", default=None, help="Ruta al fitxer GloVe (ex: glove.6B.300d.txt). Si no s'especifica, usa vocabulari estàndard.")
    p.add_argument("--glove_dim", type=int, default=300, help="Dimensió dels vectors GloVe (50, 100, 200 o 300)") # ha de coincidir amb el fitxer
    args = p.parse_args() 

    if args.glove:
        # OPCIÓ 2: vocabulari amb embeddings semàntics GloVe
        vocab = build_vocab_glove(args.captions, args.glove, args.threshold, args.glove_dim)
        print(f"Mode: GloVe (dim={args.glove_dim})")
    else:
        # OPCIÓ 1: vocabulari estàndard (índexs numèrics sense semàntica)
        vocab = build_vocab(args.captions, args.threshold)
        print("Mode: estàndard (sense GloVe)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True) # crea carpeta on guardar el vocab si no existeix
    with open(args.out, "wb") as f: # obre en mode escriptura binària
        pickle.dump(vocab, f) # guarda vocab al fitxer (inclou glove_embeddings si s'ha usat GloVe)
    print(f"Vocab size: {len(vocab)} (threshold={args.threshold})") # imprimeix mida del vocabulari i threshold utilitzat
    print(f"Saved to {args.out}") # imprimeix ruta on s'ha guardat el vocabulari


if __name__ == "__main__": # si fas import vocabulary.py des d'un altre fitxer, no s'executa main(), 
    main()                 # però si executes `python vocabulary.py` sí que s'executa
