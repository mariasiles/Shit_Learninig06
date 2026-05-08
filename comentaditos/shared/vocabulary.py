"""
vocabulary.py
=============
Construeix i gestiona el vocabulari del projecte.

Totes les xarxes neuronals treballen amb números, no amb text. Per tant,
cal convertir cada paraula a un enter (índex) i viceversa. Aquest fitxer
s'encarrega de tot això: construir la llista de paraules conegudes,
assignar-los un número, i convertir frases senceres a seqüències de
números (encode) o tornar-les a text (decode).

A més, inclou funcions per carregar embeddings preentrenats (GloVe i
Word2Vec), que permeten inicialitzar el model amb representacions de
paraules ja apreses, en lloc de partir de zero.
"""

from __future__ import annotations  # permet escriure 'list[str]' en lloc de 'List[str]'
                                     # sense importar res extra (Python 3.7+)

import argparse   # permet definir arguments de línia de comandes (--captions, --out, --threshold)
import pickle     # permet guardar/carregar objectes Python directament a disc (format binari .pkl)
import re         # expressions regulars: s'usa per trobar paraules dins d'un text
from collections import Counter  # diccionari especial que compta ocurrències automàticament
                                  # Ex: Counter(["gos","gat","gos"]) → {"gos":2, "gat":1}
from pathlib import Path          # gestió de rutes de fitxers de forma neta i multiplataforma

import pandas as pd
import torch  # llegir fitxers CSV com a taules (DataFrames)

# ─────────────────────────────────────────────────────────────────────────────
# TOKENS ESPECIALS
# Quatre "paraules" reservades que mai apareixeran com a captions reals,
# però que el model necessita per saber on comença/acaba una frase, etc.
# ─────────────────────────────────────────────────────────────────────────────

PAD   = "<pad>"    # índex 0 — Padding: emplena les frases curtes dins d'un batch perquè totes tinguin la mateixa longitud
START = "<start>"  # índex 1 — Inici de frase: el decoder sempre comença generant a partir d'aquest token
END   = "<end>"    # índex 2 — Fi de frase: quan el decoder el genera, s'atura
UNK   = "<unk>"    # índex 3 — Unknown: qualsevol paraula que no estigui al vocabulari


# ─────────────────────────────────────────────────────────────────────────────
# TOKENITZACIÓ
# ─────────────────────────────────────────────────────────────────────────────

def simple_tokenize(text: str) -> list[str]:
    """
    Converteix una frase en una llista de paraules (tokens).

    Pas 1: Posa tot en minúscules → "A Dog" → "a dog"
    Pas 2: Extreu seqüències de lletres/números/apòstrofs, ignorant
           puntuació, espais, comes, signes d'exclamació, etc.

    Exemples:
        "A dog, running!"   → ["a", "dog", "running"]
        "It's very happy."  → ["it's", "very", "happy"]
        "2 cats & 3 dogs"   → ["2", "cats", "3", "dogs"]
    """
    text   = text.lower()                    # minúscules: "Dog" i "dog" seran la mateixa paraula
    tokens = re.findall(r"[a-z0-9']+", text) # expresió regular: troba totes les seqüències de
                                             # lletres minúscules (a-z), dígits (0-9) o apòstrof (')
                                             # re.findall retorna una llista de coincidències
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    """Gestiona la correspondència bidireccional entre paraules i números.

    Conté dos diccionaris complementaris:
        word2idx: {"gos": 4, "gat": 5, ...}   → paraula  → número
        idx2word: {4: "gos", 5: "gat", ...}   → número   → paraula

    Flux típic d'ús:
        vocab = build_vocab("captions.txt")          # construeix el vocabulari
        ids   = vocab.encode("a dog runs")           # [1, 4, 23, 87, 2]
        text  = vocab.decode([1, 4, 23, 87, 2])      # "a dog runs"
    """

    def __init__(self):
        self.word2idx: dict[str, int] = {}  # paraula → índex enter
        self.idx2word: dict[int, str] = {}  # índex enter → paraula
        self.idx = 0                        # comptador intern: pròxim índex disponible

        # Afegim els 4 tokens especials en ordre fix: PAD=0, START=1, END=2, UNK=3
        # És IMPORTANT que PAD sigui el 0, perquè els tensors de padding s'inicialitzen a 0
        for tok in (PAD, START, END, UNK):
            self.add_word(tok)

    def add_word(self, word: str) -> None:
        """
        Afegeix una paraula nova al vocabulari (si ja hi és, no fa res).

        Actualitza els dos diccionaris i incrementa el comptador d'índex.
        """
        if word not in self.word2idx:          # comprova que la paraula no existeix ja
            self.word2idx[word] = self.idx     # assigna l'índex actual a la paraula
            self.idx2word[self.idx] = word     # relació inversa: índex → paraula
            self.idx += 1                      # prepara el comptador per a la pròxima paraula

    def __call__(self, word: str) -> int:
        """
        Permet usar vocab("gos") en lloc de vocab.word2idx.get("gos").

        Si la paraula no existeix al vocabulari, retorna l'índex de <unk> (3).
        Això evita errors quan el model troba paraules noves durant la inferència.
        """
        return self.word2idx.get(word, self.word2idx[UNK])

    def __len__(self) -> int:
        """Retorna el nombre total de paraules al vocabulari (inclou tokens especials)."""
        return len(self.word2idx)

    def encode(self, caption: str, add_special: bool = True) -> list[int]:
        """
        Converteix una frase de text en una llista d'índexs enters.

        Exemple:
            vocab.encode("a dog runs")
            → tokenitza:  ["a", "dog", "runs"]
            → indexa:     [8, 4, 23]
            → amb specials: [1, 8, 4, 23, 2]   ← [<start>, ..., <end>]

        Args:
            caption:     frase de text a codificar
            add_special: si True, afegeix <start> al principi i <end> al final
        """
        tokens = simple_tokenize(caption)      # "a dog runs" → ["a", "dog", "runs"]
        ids    = [self(t) for t in tokens]     # cada token → índex (o 3 si és <unk>)
        if add_special:
            ids = [self(START)] + ids + [self(END)]  # [1] + [8, 4, 23] + [2] = [1, 8, 4, 23, 2]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        Converteix una llista d'índexs enters de tornada a text llegible.

        Exemple:
            vocab.decode([1, 8, 4, 23, 2, 0, 0])
            → ignora <start>, <pad>
            → s'atura a <end>
            → retorna: "a dog runs"

        Args:
            ids:          llista d'enters (índexs del vocabulari)
            skip_special: si True, omet <pad> i <start>, i s'atura a <end>
        """
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), UNK)      # índex → paraula (o "<unk>" si no existeix)

            if skip_special and w in (PAD, START):  # <pad> i <start> s'ometen sempre
                continue
            if skip_special and w == END:            # <end> indica fi de frase → parem
                break

            words.append(w)

        return " ".join(words)  # ["a", "dog", "runs"] → "a dog runs"


# ─────────────────────────────────────────────────────────────────────────────
# CÀRREGA D'EMBEDDINGS PREENTRENATS
# ─────────────────────────────────────────────────────────────────────────────

def load_glove_weights(glove_path: str | Path, vocab: Vocabulary) -> tuple["torch.Tensor", int]:
    """
    Carrega vectors GloVe i construeix una matriu de pesos per al vocabulari.

    GloVe (Global Vectors for Word Representation) és un conjunt de vectors
    preentrenats que representen paraules com a punts en un espai numèric,
    de manera que paraules amb significat similar estan a prop.
    Entrenat amb milers de milions de paraules de Wikipedia i altres fonts.

    Retorna:
        weights:   matriu [vocab_size, glove_dim] on cada fila és el vector d'una paraula
        glove_dim: dimensió dels vectors (50, 100, 200 o 300 depenent del fitxer)

    Per a paraules del vocabulari que NO existeixen a GloVe:
        → s'inicialitzen amb valors aleatoris molt petits (randn * 0.01)
        → l'excepció és <pad>, que sempre és el vector zero

    Descàrrega: https://nlp.stanford.edu/data/glove.6B.zip
    Recomanat:  glove.6B.300d.txt  (300 dimensions, 400.000 paraules)
    """
    import torch

    print(f"[glove] carregant {glove_path}...")

    # Llegim el fitxer GloVe línia per línia.
    # Cada línia és: "paraula  0.123  -0.456  0.789  ..."
    glove: dict[str, list[float]] = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts      = line.split()          # separa per espais: ["paraula", "0.1", "-0.4", ...]
            glove[parts[0]] = [float(x) for x in parts[1:]]    # parts[0] = paraula, parts[1:] = vector numèric

    glove_dim = len(next(iter(glove.values())))  # detecta la dimensió llegint el primer vector
                                                  # next(iter(...)) → agafa el primer element del dict
    found, total = 0, len(vocab)

    # Inicialitzem la matriu de pesos amb valors aleatoris molt petits.
    # Les files on trobem la paraula a GloVe les sobreescriurem després.
    weights    = torch.randn(total, glove_dim) * 0.01  # [vocab_size, glove_dim]
    weights[0] = 0  # índex 0 = <pad> → vector zero (no té significat semàntic)

    # Omplim les files de les paraules que SÍ existeixen a GloVe
    for word, idx in vocab.word2idx.items():
        if word in glove:
            weights[idx] = torch.tensor(glove[word])  # copia el vector GloVe a la fila corresponent
            found += 1

    print(f"[glove] {found}/{total} paraules del vocabulari trobades a GloVe ({glove_dim}d)")
    return weights, glove_dim


def load_word2vec_weights(w2v_path: str | Path, vocab: Vocabulary, binary: bool | None = None) -> tuple["torch.Tensor", int]:
    """
    Carrega vectors Word2Vec i construeix una matriu de pesos per al vocabulari.

    Word2Vec (Google, 2013) és una altra família de vectors de paraules,
    entrenats amb una xarxa neuronal per predir paraules a partir del seu context.
    El model més conegut és GoogleNews (300d, 3 milions de paraules).

    Formats suportats:
        - Binari (.bin):  GoogleNews-vectors-negative300.bin  (carrega ràpid, fitxer gran)
        - Text   (.txt):  capçalera "vocab_size dim" seguida d'una paraula per línia

    Si 'binary' és None, es detecta automàticament per l'extensió del fitxer.

    Requereix: pip install gensim
    """

    import torch
    from gensim.models import KeyedVectors  # llibreria per llegir fitxers Word2Vec

    w2v_path = Path(w2v_path)
    if binary is None:
        binary = (w2v_path.suffix == ".bin")  # auto-detecta format pel extensió

    print(f"[word2vec] carregant {w2v_path} (binary={binary})...")
    wv = KeyedVectors.load_word2vec_format(str(w2v_path), binary=binary)
    # wv és un diccionari-like: wv["dog"] retorna el vector numpy de "dog"

    w2v_dim = wv.vector_size   # dimensió dels vectors (normalment 300)
    found, total = 0, len(vocab)

    # Matriu inicialitzada aleatòriament per a les paraules no trobades
    weights    = torch.randn(total, w2v_dim) * 0.01
    weights[0] = 0  # <pad> → vector zero

    for word, idx in vocab.word2idx.items():
        if word in wv:
            weights[idx] = torch.tensor(wv[word])  # copia vector Word2Vec
            found += 1

    print(f"[word2vec] {found}/{total} paraules del vocabulari trobades a Word2Vec ({w2v_dim}d)")
    return weights, w2v_dim


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓ DEL VOCABULARI
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(captions_csv: str | Path, threshold: int = 5) -> Vocabulary:
    """
    Llegeix totes les captions i construeix el vocabulari.

    Procés:
        1. Llegir el CSV de captions
        2. Tokenitzar cada caption i comptar la freqüència de cada paraula
        3. Afegir al vocabulari únicament les paraules que apareixen ≥ threshold vegades

    Per què el threshold? Paraules molt rares (typos, noms propis únics, etc.)
    no aporten informació útil al model i n'augmenten la mida innecessàriament.
    Amb threshold=5 i Flickr8k, el vocabulari queda al voltant de ~3.000 paraules.

    Args:
        captions_csv: ruta al fitxer CSV de captions
        threshold:    nombre mínim d'aparicions per incloure una paraula (defecte: 5)
    """
    df = pd.read_csv(captions_csv)

    counter: Counter[str] = Counter()       # comptador de freqüències buit
    for cap in df["caption"].astype(str):   # itera sobre totes les captions del CSV
        counter.update(simple_tokenize(cap))  # tokenitza i actualitza les freqüències
        # Ex: simple_tokenize("a dog") → ["a","dog"]
        # counter.update(["a","dog"]) → counter["a"]+=1, counter["dog"]+=1

    vocab = Vocabulary()                    # vocabulari buit (ja té els 4 tokens especials)
    for word, count in counter.items():
        if count >= threshold:              # només paraules prou freqüents
            vocab.add_word(word)

    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# PUNT D'ENTRADA PER TERMINAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Executa la construcció del vocabulari des del terminal.

    Ús:
        python -m src.shared.vocabulary --captions dataset/captions.txt --out dataset/vocab.pkl
        python -m src.shared.vocabulary --threshold 3   ← vocabulari més gran (menys estricte)
    """
    p = argparse.ArgumentParser(description="Construeix i guarda el vocabulari.")
    p.add_argument("--captions",  default="data/flickr8k/captions.txt",
                   help="Ruta al CSV de captions")
    p.add_argument("--out",       default="data/flickr8k/vocab.pkl",
                   help="On guardar el vocabulari (format .pkl)")
    p.add_argument("--threshold", type=int, default=5,
                   help="Freqüència mínima per incloure una paraula")
    args = p.parse_args()

    vocab = build_vocab(args.captions, args.threshold)

    # Crea la carpeta de destí si no existeix (mkdir -p equivalent)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Guarda l'objecte Vocabulary complet en format binari
    # pickle.dump serialitza l'objecte Python i el guarda directament
    with open(args.out, "wb") as f:   # "wb" = write binary
        pickle.dump(vocab, f)

    print(f"Vocab size: {len(vocab)} (threshold={args.threshold})")
    print(f"Saved to {args.out}")


# Guarda de protecció: aquest bloc NOMÉS s'executa si el fitxer es llança
# directament (`python vocabulary.py`). Si s'importa des d'un altre fitxer
# (`from vocabulary import Vocabulary`), el main() NO s'executa.
if __name__ == "__main__":
    main()