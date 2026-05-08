"""
train.py  —  Script d'entrenament complet (src/attention/train.py)
==================================================================
Versió avançada que incorpora totes les millores de la iteració actual:

  Novetats respecte a la versió anterior:
  ────────────────────────────────────────
  1. --no-semantic-loss:      desactiva la SemanticCrossEntropy fins i tot
                              amb GloVe (permet usar CE estàndard + embeddings preentrenats,
                              que és la configuració guanyadora dels experiments)
  2. --finetune-cnn-epoch:    desbloqueja la última capa de la ResNet (layer4) a
                              partir d'una época determinada → fine-tuning de la CNN
  3. --ds-lambda:             regularització Doubly Stochastic Attention
                              (penalitza que el model no miri totes les regions)
  4. --label-smoothing:       suavitzat d'etiquetes per a CrossEntropyLoss
                              (redueix l'overfitting en classificació)
  5. --resume-from:           permet continuar un entrenament interromput
  6. --flickr30k-hf:          suport per al dataset Flickr30k via HuggingFace
  7. SCST (Self-Critical Sequence Training):
       --scst-epochs, --scst-lr, etc.
       Fine-tuning addicional que optimitza directament la mètrica CIDEr-D
       en lloc de la Cross-Entropy.
  8. safe_save():             guarda checkpoints de forma segura (reintents NFS)

Ús des del terminal:
    # Configuració guanyadora (atenció + GloVe + CE estàndard):
    python -m src.attention.train \\
        --glove-path glove.6B.300d.txt \\
        --no-semantic-loss \\
        --backbone resnet152 \\
        --epochs 15 \\
        --patience 5

    # Amb fine-tuning de la CNN a partir de l'época 5:
    python -m src.attention.train --finetune-cnn-epoch 5

    # Continuant un entrenament interromput:
    python -m src.attention.train --resume-from checkpoints_attention/ckpt_epoch8.pt

    # Amb SCST (CIDEr-D reward):
    python -m src.attention.train --epochs 15 --scst-epochs 5 --scst-lr 5e-5

    # Dataset Flickr30k (HuggingFace):
    python -m src.attention.train --flickr30k-hf --epochs 20
"""

from __future__ import annotations

import argparse
import contextlib
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score  import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

from src.shared.dataset    import (
    get_loaders, get_scst_loader, split_image_ids, load_captions_df,
    get_loaders_hf, get_scst_loader_hf,
)
from src.attention.model   import AttentionDecoder, EncoderCNNAttention
from src.attention.sample  import caption_image, caption_pil_image
from src.shared.vocabulary import (
    Vocabulary, build_vocab, build_vocab_hf, simple_tokenize,
    load_glove_weights, load_word2vec_weights,
)
from src.shared.losses import SemanticCrossEntropyLoss, build_soft_labels


# ═════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    """Defineix tots els hiperparàmetres configurables des del terminal."""
    p = argparse.ArgumentParser()

    # ── Rutes ─────────────────────────────────────────────────────────────────
    p.add_argument("--images-dir",      default="dataset/Images")
    p.add_argument("--captions-csv",    default="dataset/captions.txt")
    p.add_argument("--vocab-path",      default="dataset/vocab.pkl")
    p.add_argument("--checkpoints-dir", default="checkpoints_attention")
    p.add_argument("--vocab-threshold", type=int, default=5)

    # ── Arquitectura ──────────────────────────────────────────────────────────
    p.add_argument("--embed-size",    type=int,   default=256)
    p.add_argument("--hidden-size",   type=int,   default=512)
    p.add_argument("--attention-dim", type=int,   default=256)
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--backbone",      default="resnet152",
                   help="'resnet50' (ràpid) o 'resnet152' (millors resultats, usat als experiments).")

    # ── Entrenament ───────────────────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--num-workers", type=int,   default=2)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--log-step",    type=int,   default=20)

    # ── Embeddings preentrenats ────────────────────────────────────────────────
    p.add_argument("--glove-path",      default=None,
                   help="Ruta al fitxer GloVe (.txt). Activa embeddings GloVe.")
    p.add_argument("--word2vec-path",   default=None,
                   help="Ruta al fitxer Word2Vec (.bin o .txt).")
    p.add_argument("--word2vec-binary", action="store_true",
                   help="Força lectura binària del Word2Vec.")
    p.add_argument("--no-semantic-loss", action="store_true",
                   help="NOVETAT: usa CrossEntropyLoss estàndard fins i tot amb GloVe/Word2Vec. "
                        "Permet separar l'efecte dels embeddings de l'efecte de la loss semàntica. "
                        "La SemanticCrossEntropyLoss ha mostrat disparar la perplexitat als experiments; "
                        "aquesta opció permet usar GloVe sense la loss semàntica.")
    p.add_argument("--freeze-embeddings", action="store_true",
                   help="Si s'activa, els pesos d'embedding NO s'actualitzen. "
                        "Millors resultats amb freeze=False (fine-tuning dels embeddings).")
    p.add_argument("--semantic-temp", type=float, default=10.0,
                   help="Temperatura pels soft labels semàntics.")

    # ── Fine-tuning de la CNN ──────────────────────────────────────────────────
    p.add_argument("--finetune-cnn-epoch", type=int, default=None,
                   help="NOVETAT: época a partir de la qual es desbloqueja layer4 de la ResNet. "
                        "None = mai (CNN sempre congelada). Ex: --finetune-cnn-epoch 5 "
                        "desbloqueja layer4 a partir de l'época 5 amb lr/10.")

    # ── Regularització avançada ────────────────────────────────────────────────
    p.add_argument("--ds-lambda", type=float, default=1.0,
                   help="NOVETAT: pes de la regularització Doubly Stochastic Attention. "
                        "0 = desactivada. 1.0 = activada (per defecte). "
                        "Força el model a atendre totes les regions de la imatge.")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="NOVETAT: label smoothing per CrossEntropyLoss. "
                        "0.0 = desactivat (defecte). 0.1 = recomanat si s'activa. "
                        "Redueix l'overfitting evitant que el model sigui massa confiat.")

    # ── Represa d'entrenament ──────────────────────────────────────────────────
    p.add_argument("--resume-from", default=None,
                   help="NOVETAT: ruta a un checkpoint des del qual continuar l'entrenament. "
                        "Útil si l'entrenament s'ha interromput. "
                        "Ex: --resume-from checkpoints_attention/ckpt_epoch8.pt")

    # ── Dataset HuggingFace (Flickr30k) ────────────────────────────────────────
    p.add_argument("--flickr30k-hf", action="store_true",
                   help="NOVETAT: usa el dataset Flickr30k de HuggingFace "
                        "(nlphuji/flickr30k) en lloc del CSV local. "
                        "Requereix: pip install datasets")
    p.add_argument("--flickr30k-hf-cache", default="dataset/flickr30k_hf",
                   help="Carpeta cache del dataset HuggingFace.")

    # ── WandB ─────────────────────────────────────────────────────────────────
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb-project", default="image-captioning")
    p.add_argument("--wandb-entity",  default=None)
    p.add_argument("--run-name",      default=None)

    # ── SCST (Self-Critical Sequence Training) ─────────────────────────────────
    p.add_argument("--scst-epochs", type=int, default=0,
                   help="NOVETAT: èpoques de fine-tuning SCST amb CIDEr-D reward. "
                        "0 = sense SCST (defecte). Recomanat: 3-5 èpoques après del CE.")
    p.add_argument("--scst-lr", type=float, default=5e-5,
                   help="Taxa d'aprenentatge per a SCST. Ha de ser molt baixa (1e-5 a 1e-4). "
                        "Un lr massa alt destruirà el model entrenat prèviament.")
    p.add_argument("--scst-batch-size", type=int, default=16,
                   help="Batch size per a SCST (recomanat <= CE batch size). "
                        "SCST requereix 2 forward passes per imatge (sample + greedy) "
                        "i és molt més intensiu en memòria que l'entrenament CE.")
    p.add_argument("--scst-max-len", type=int, default=20,
                   help="Longitud màxima de les seqüències mostrejades durant SCST.")
    p.add_argument("--scst-checkpoint", default=None,
                   help="Checkpoint inicial per SCST. Si s'especifica amb --epochs 0, "
                        "salta la fase CE i fa SCST directament des d'aquest checkpoint.")
    p.add_argument("--scst-checkpoints-dir", default="checkpoints_attention_scst",
                   help="Carpeta on guardar els checkpoints de la fase SCST.")

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
# UTILITATS
# ═════════════════════════════════════════════════════════════════════

def safe_save(obj, path, retries: int = 5):
    """Guarda un objecte amb torch.save de forma segura, amb reintents.

    Per a qué serveix?
        En sistemes amb carpetes de xarxa (NFS, Google Drive muntat, etc.),
        escriure fitxers grans pot fallar aleatòriament per errors transitoris
        de la xarxa. safe_save fa fins a 'retries' intents abans de rendir-se.

    Estratègia atòmica (per evitar checkpoints corruptes):
        1. Guarda primer a un fitxer temporal (.tmp_ckpt_epoch5.pt)
        2. Si té èxit, mou el fitxer temporal al nom definitiu (ckpt_epoch5.pt)
        El mou és atòmic en la majoria de sistemes de fitxers:
        o la imatge antiga o la nova, mai un estat intermedi corrupte.

    Args:
        obj:     l'objecte a guardar (el diccionari del checkpoint)
        path:    ruta de destí definitiva
        retries: nombre màxim d'intents
    """
    import shutil, time
    path = Path(path)
    for attempt in range(retries):
        try:
            tmp = path.parent / f".tmp_{path.name}"  # fitxer temporal ocult
            torch.save(obj, tmp)                      # guarda al temporal
            shutil.move(str(tmp), str(path))          # mou al definitiu (atòmic)
            return
        except RuntimeError:
            if attempt < retries - 1:
                print(f"[ckpt] error NFS (intent {attempt+1}/{retries}), reintentant...")
                time.sleep(3)  # espera 3 segons abans de tornar a intentar
            else:
                raise  # si s'esgoten els intents, llença l'error


def get_or_build_vocab(args) -> Vocabulary:
    """Carrega el vocabulari des del disc o el construeix si no existeix."""
    vp = Path(args.vocab_path)
    if vp.exists():
        with open(vp, "rb") as f:
            return pickle.load(f)
    vocab = build_vocab(args.captions_csv, threshold=args.vocab_threshold)
    vp.parent.mkdir(parents=True, exist_ok=True)
    with open(vp, "wb") as f:
        pickle.dump(vocab, f)
    print(f"[vocab] built and saved to {vp} (size={len(vocab)})")
    return vocab


@torch.no_grad()
def evaluate(encoder, decoder, loader, criterion, device) -> float:
    """Calcula la pèrdua mitjana sobre el conjunt de validació.

    Diferència respecte a la versió anterior:
        decoder() ara retorna (outputs, alphas_sum) en lloc de només outputs.
        El segon valor (alphas_sum) és la suma dels pesos d'atenció per a la
        regularització Doubly Stochastic. Aquí l'ignorem (usem _).
    """
    encoder.eval()
    decoder.eval()
    losses = []
    for images, captions, lengths in loader:
        images   = images.to(device)
        captions = captions.to(device)
        targets  = pack_padded_sequence(
            captions[:, 1:], [l - 1 for l in lengths], batch_first=True
        ).data
        features = encoder(images)
        outputs, _ = decoder(features, captions, lengths)
        # _ = alphas_sum (ignorat durant la validació, no cal regularitzar)
        losses.append(criterion(outputs, targets).item())
    return float(np.mean(losses))


# ═════════════════════════════════════════════════════════════════════
# HELPERS PER A CIDEr-D (mètrica per a SCST)
# ═════════════════════════════════════════════════════════════════════

def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Extreu tots els n-grams d'una seqüència de tokens i els compta.

    Un n-gram és una seqüència de n paraules consecutives.
    Exemple amb n=2 (bigrames):
        tokens = ["a", "dog", "runs", "fast"]
        bigrams = [("a","dog"), ("dog","runs"), ("runs","fast")]
        → Counter({("a","dog"):1, ("dog","runs"):1, ("runs","fast"):1})

    Implementació eficient amb zip:
        zip(*[tokens[i:] for i in range(n)]) genera totes les seqüències de n tokens.
        Per n=2: zip(tokens[0:], tokens[1:]) = parelles consecutives.
        Counter compta les ocurrències de cada n-gram.

    Args:
        tokens: llista de strings (caption tokenitzada)
        n:      mida del n-gram (1=unigrames, 2=bigrames, 3=trigrames, 4=4-grames)

    Returns:
        Counter amb el nombre d'ocurrències de cada n-gram.
    """
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def compute_idf_weights(df_train: "pd.DataFrame", n_max: int = 4) -> dict:
    """Pre-calcula els pesos IDF dels n-grams des de les captions d'entrenament.

    Per a qué serveix?
        CIDEr-D és una mètrica basada en TF-IDF per a n-grams.
        IDF (Inverse Document Frequency) mesura la rareza d'un n-gram:
        un n-gram que apareix en poques imatges té IDF alt (és informatiu),
        un que apareix en gairebé totes les imatges té IDF baix (és genèric).
        Ex: "a" té IDF baix (apareix a gairebé totes les captions)
            "skateboard" té IDF alt (apareix en poques captions)

    Fórmula IDF (Laplace-smoothed):
        IDF(ngram) = log( (N+1) / (df(ngram)+1) )
        On N = nombre d'imatges, df = en quantes imatges apareix el ngram.

    S'executa UNA SOLA VEGADA abans del bucle SCST (O(|train|) en temps).
    Emmagatzemar els IDF en un diccionari permet consultes O(1) durant SCST.

    Args:
        df_train: DataFrame amb les captions d'entrenament (columnes "image", "caption")
        n_max:    ordre màxim dels n-grams (defecte 4, com a l'article CIDEr-D)

    Returns:
        dict {n: {ngram_tuple: idf_float}} per n en 1..n_max
    """
    images = df_train["image"].unique()  # llista d'imatges úniques del train
    N      = len(images)                 # nombre total d'imatges (per a la fórmula IDF)

    idf: dict[int, dict] = {}
    for n in range(1, n_max + 1):
        df_cnt: dict = defaultdict(int)  # comptador de "en quantes imatges apareix cada n-gram"
        for img in images:
            caps = df_train[df_train["image"] == img]["caption"].tolist()
            seen: set = set()  # n-grams únics per a AQUESTA imatge (un ngram compta una sola vegada per imatge)
            for cap in caps:
                for ng in _get_ngrams(simple_tokenize(str(cap)), n):
                    seen.add(ng)
            for ng in seen:
                df_cnt[ng] += 1  # incrementa el comptador d'imatges on apareix aquest ngram

        # IDF suavitzat de Laplace per evitar log(0) i suavitzar ngrams rars
        idf[n] = {ng: math.log((N + 1) / (cnt + 1)) for ng, cnt in df_cnt.items()}

    print(f"[CIDEr-D] IDF computed from {N} training images")
    return idf


def cider_d_score(
    hyp_tokens:      list[str],
    ref_tokens_list: list[list[str]],
    idf:             dict,
    n_max:           int = 4,
) -> float:
    """Calcula el score CIDEr-D per a una hipòtesi vs múltiples referències.

    Concepte CIDEr-D (Consensus-based Image Description Evaluation - Discriminative):
        Mesura la similitud cosinus entre els vectors TF-IDF dels n-grams
        de la hipòtesi i les referències. A diferència de BLEU (que només
        compta coincidències exactes), CIDEr-D pondera els n-grams per IDF:
        encertar un n-gram rar val molt més que encertar un de comú.

        Puntuació final = promig sobre ordres 1..n_max, escalada per 10.

    CIDEr-D clipping (vs CIDEr bàsic):
        El count de cada n-gram a la hipòtesi es limita ("clipa") al màxim
        que apareix en qualsevol referència. Penalitza la repetició:
        si el model genera "dog dog dog", el clip evita que puntui bé
        per tenir molts unigrames "dog".

    TF (Term Frequency):
        tf(ngram, text) = count(ngram, text) / len(text)
        Normalitza per la longitud per no afavorir textos llargs.

    Cosinus:
        sim = (h·r) / (|h|·|r|)
        Mesura l'angle entre els vectors TF-IDF de la hipòtesi i la referència.
        1.0 = idèntics, 0.0 = cap n-gram en comú.

    Args:
        hyp_tokens:      caption generada (tokenitzada)
        ref_tokens_list: llista de captions de referència (tokenitzades)
        idf:             pesos IDF precalculats per compute_idf_weights()
        n_max:           ordre màxim de n-grams (defecte 4)

    Returns:
        score CIDEr-D ∈ [0, ~10] (les millors captions humanes puntuen ~10)
    """
    if not hyp_tokens or not ref_tokens_list:
        return 0.0

    total = 0.0
    for n in range(1, n_max + 1):
        idf_n    = idf.get(n, {})                  # IDF per a n-grams d'ordre n
        h_ngrams = _get_ngrams(hyp_tokens, n)       # n-grams de la hipòtesi amb el seu count

        ref_ngrams_list = [_get_ngrams(ref, n) for ref in ref_tokens_list]
        # llista de Counters: un per cada referència

        if not h_ngrams:
            continue  # si la hipòtesi no té cap n-gram d'ordre n, contribució 0

        # ── CIDEr-D clipping ──────────────────────────────────────────────────
        # Per a cada n-gram de la hipòtesi, limitem el seu count al màxim que
        # apareix en qualsevol referència. Penalitza repeticions innecessàries.
        h_clipped = {
            ng: min(cnt, max((r.get(ng, 0) for r in ref_ngrams_list), default=0))
            for ng, cnt in h_ngrams.items()
        }

        len_h = max(len(hyp_tokens) - n + 1, 1)  # nombre de posicions possibles per a n-grams d'ordre n

        # ── Norma del vector TF-IDF de la hipòtesi ────────────────────────────
        h_norm_sq = sum(
            ((h_clipped.get(ng, 0) / len_h) * idf_n.get(ng, 0.0)) ** 2
            for ng in h_clipped
        )
        h_norm = math.sqrt(h_norm_sq + 1e-10)  # +1e-10 evita divisió per zero

        # ── Similitud cosinus vs cada referència ──────────────────────────────
        ref_sum = 0.0
        for ref_tokens, ref_ngrams in zip(ref_tokens_list, ref_ngrams_list):
            len_r = max(len(ref_tokens) - n + 1, 1)

            # Producte escalar TF-IDF(hipòtesi) · TF-IDF(referència)
            dot = sum(
                (h_clipped[ng] / len_h) * idf_n.get(ng, 0.0)
                * (ref_ngrams[ng] / len_r) * idf_n.get(ng, 0.0)
                for ng in h_clipped
                if ng in ref_ngrams  # només n-grams que apareixen a ambdues
            )

            # Norma del vector TF-IDF de la referència
            r_norm = math.sqrt(sum(
                ((cnt / len_r) * idf_n.get(ng, 0.0)) ** 2
                for ng, cnt in ref_ngrams.items()
            ) + 1e-10)

            ref_sum += dot / (h_norm * r_norm)  # similitud cosinus per a aquesta referència

        total += ref_sum / len(ref_tokens_list)  # promig sobre les references per a ordre n

    return total * 10.0 / n_max
    # Escala per 10 (convenció de CIDEr-D) i divideix pel nombre d'ordres (promig)


# ═════════════════════════════════════════════════════════════════════
# SCST (Self-Critical Sequence Training)
# ═════════════════════════════════════════════════════════════════════

def scst_epoch(
    encoder:       "EncoderCNNAttention",
    decoder:       "AttentionDecoder",
    loader,
    optimizer:     "torch.optim.Optimizer",
    vocab:         "Vocabulary",
    refs_by_image: dict,
    idf:           dict,
    device:        "torch.device",
    args,
    use_wandb:     bool = False,
    epoch:         int  = 0,
) -> float:
    """Executa una época d'entrenament SCST (Self-Critical Sequence Training).

    Concepte SCST (Rennie et al., 2017):
        L'entrenament CE (Cross-Entropy) minimitza la probabilitat d'equivocar-se
        token per token, però la mètrica real (CIDEr-D) depèn de la seqüència
        COMPLETA. Hi ha un buit entre la loss d'entrenament i la mètrica d'avaluació.

        SCST tanca aquest buit optimitzant directament CIDEr-D usant REINFORCE
        (un algoritme de policy gradient de l'aprenentatge per reforç):

            Per a cada imatge:
              1. Generem una seqüència MOSTREJADA (sample) — exploració estocàstica
              2. Generem una seqüència GREEDY (determinista) — la "línia base"
              3. Reward = CIDEr(sample) - CIDEr(greedy)
                 Si el sample és millor que greedy → reward positiu → reforcem
                 Si el sample és pitjor → reward negatiu → desreforcem

            Loss REINFORCE: -reward · sum(log_probs dels tokens del sample)
            Si reward > 0: incrementem la probabilitat del sample (bon camí)
            Si reward < 0: la decrementem (mal camí)

    Normalització per batch:
        Els rewards es normalitzen per batch (zero-mean, unit-std).
        Redueix la variança del gradient (problema clàssic de REINFORCE)
        i estabilitza l'entrenament.

    Gradient clipping:
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        Limita la magnitud del gradient per evitar explosions (gradient explosion).

    Args:
        encoder:       l'encoder (sempre congelat durant SCST)
        decoder:       el decoder (s'entrena)
        loader:        DataLoader especial per a SCST (inclou img_ids)
        optimizer:     optimitzador Adam amb lr baix (scst_lr)
        vocab:         vocabulari
        refs_by_image: dict {img_id: [llista de captions de referència]}
        idf:           pesos IDF precalculats per compute_idf_weights()
        device:        cpu o cuda
        args:          arguments de la comanda
        use_wandb:     si registrar a WandB
        epoch:         número d'época SCST (per al log)

    Returns:
        reward mig de l'época (float) — mesura la millora respecte al greedy
    """
    encoder.eval()   # encoder SEMPRE congelat durant SCST (estabilitat)
    decoder.train()  # decoder s'actualitza

    start_idx = vocab.word2idx["<start>"]
    end_idx   = vocab.word2idx["<end>"]

    total_reward = 0.0
    n_batches    = 0

    for batch_idx, (images, captions, lengths, img_ids) in enumerate(loader):
        # img_ids: llista de noms de fitxer de les imatges del batch (ex: ["dog.jpg", "cat.jpg", ...])
        # Necessari per accedir a les references de cada imatge (refs_by_image[img_id])

        images = images.to(device, non_blocking=True)

        with torch.no_grad():
            features = encoder(images)
            # L'encoder està congelat. torch.no_grad() evita que es calculin
            # gradients per a ell (estalvi de memòria i temps).
            # features: [B, 49, 2048]

        # ── Generació del sample i del greedy ──────────────────────────────────

        sampled_tokens_list, log_probs_list = decoder.sample_batch_with_logprobs(
            features, start_idx, end_idx, max_len=args.scst_max_len
        )
        # sample_batch_with_logprobs: genera una seqüència ESTOCÀSTICA per a cada imatge.
        # "Estocàstica" significa que en lloc de prendre sempre el token màxim (greedy),
        # MOSTREGT un token de la distribució de probabilitats.
        # Retorna:
        #   sampled_tokens_list: llista de llistes d'enters (un per imatge) — els tokens generats
        #   log_probs_list:      llista de tensors — log-probabilitat de cada token generat
        # Necessitem log_probs per calcular el gradient REINFORCE.

        greedy_tokens_list = decoder.greedy_batch(
            features, start_idx, end_idx, max_len=args.scst_max_len
        )
        # greedy_batch: genera una seqüència DETERMINISTA (sempre el token màxim).
        # S'usa com a "baseline" per al reward diferencial: reward = CIDEr(sample) - CIDEr(greedy).
        # Usar el greedy com a baseline (auto-crític) és la clau de SCST:
        # si el sample és millor que el greedy, hem d'encoratjar-lo; si és pitjor, desalentar-lo.

        # ── Càlcul del reward per imatge ──────────────────────────────────────

        raw_rewards:  list[float]          = []
        log_probs_all: list[torch.Tensor]  = []

        for i, img_id in enumerate(img_ids):
            refs = refs_by_image.get(img_id, [])
            if not refs:
                continue  # si no hi ha referencias per a aquesta imatge, la saltem

            refs_tok = [simple_tokenize(str(r)) for r in refs]
            # Tokenitzem totes les referencias de la imatge

            sampled_words = [vocab.idx2word.get(t, "<unk>") for t in sampled_tokens_list[i]]
            greedy_words  = [vocab.idx2word.get(t, "<unk>") for t in greedy_tokens_list[i]]
            # Convertim els índexs a paraules per a CIDEr-D

            r_sample = cider_d_score(sampled_words, refs_tok, idf)
            r_greedy  = cider_d_score(greedy_words,  refs_tok, idf)
            # CIDEr-D per al sample generat i per al greedy de referència

            raw_rewards.append(r_sample - r_greedy)
            # Reward diferencial: positiu si el sample supera el greedy, negatiu si és pitjor

            log_probs_all.append(log_probs_list[i])
            # Guardem els log-probs del sample per al gradient REINFORCE

        if not raw_rewards:
            continue  # batch buit (totes les imatges sense referencies), saltem

        # ── Normalització del reward per batch ────────────────────────────────
        rewards_t = torch.tensor(raw_rewards, dtype=torch.float32, device=device)
        if rewards_t.std() > 1e-6:
            rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
        # Normalitza a zero-mean, unit-std dins del batch.
        # Redueix la variança del gradient REINFORCE (que és notòriament alta).
        # 1e-6 és un llindar: si tots els rewards són iguals (std≈0), no normalitzem
        # per evitar dividir per quasi-zero.

        # ── Loss REINFORCE i backpropagation ──────────────────────────────────
        loss = sum(-r * lp.sum() for r, lp in zip(rewards_t, log_probs_all))
        loss = loss / len(raw_rewards)
        # Fórmula REINFORCE: L = -E[reward · sum(log_prob)]
        # Intuïció:
        #   Si reward > 0 (sample millor que greedy) → volem maximitzar log_prob
        #   → el signe negatiu converteix la maximització en minimització (per backprop)
        #   Si reward < 0 (sample pitjor que greedy) → volem minimitzar log_prob
        #   → el signe negatiu ho converteix en maximització, i minimitzem la negative
        # Dividim per n per normalitzar i que el lr sigui independent del batch size.

        optimizer.zero_grad()
        loss.backward()
        # Calcula els gradients de la loss REINFORCE respecte als pesos del decoder.

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        # Gradient clipping: si la norma L2 de tots els gradients supera 5.0,
        # els escala tots proporcionalment perquè la norma sigui exactament 5.0.
        # Evita explosions del gradient que destruirien el model entrenat prèviament.
        # (El gradient de REINFORCE pot tenir alta variança → especialment important aquí)

        optimizer.step()
        # Actualitza els pesos del decoder en la direcció que incrementa CIDEr-D.

        mean_r = float(torch.tensor(raw_rewards).mean())
        total_reward += mean_r
        n_batches    += 1

        if batch_idx % 20 == 0:
            print(f"  [SCST] epoch {epoch}  batch {batch_idx}/{len(loader)}"
                  f"  mean_reward={mean_r:.4f}")
            if use_wandb:
                import wandb
                wandb.log({"scst/mean_reward": mean_r, "scst/epoch": epoch})

    return total_reward / max(n_batches, 1)
    # Reward mig de l'época. Si n_batches=0 (epoch buida), retornem 0.


# ═════════════════════════════════════════════════════════════════════
# FUNCIÓ PRINCIPAL
# ═════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)


    # ─── 1. DATASET I VOCABULARI ───────────────────────────────────────────────

    if args.flickr30k_hf:
        # ── Mode HuggingFace (Flickr30k) ─────────────────────────────────────
        # Carrega el dataset directament de HuggingFace sense necessitar els fitxers locals.
        # Requereix connexió a Internet la primera vegada (es baixa ~10GB).
        from datasets import load_dataset
        print("[data] carregant Flickr30k HuggingFace...")
        hf_ds = load_dataset(
            "nlphuji/flickr30k",
            trust_remote_code=True,
            cache_dir=args.flickr30k_hf_cache,
        )
        # hf_ds és un DatasetDict amb splits ("test" — Flickr30k no té splits estàndard,
        # però té un camp "split" per a cada exemple: "train", "val", "test").

        vp = Path(args.vocab_path)
        if vp.exists():
            with open(vp, "rb") as f:
                import pickle; vocab = pickle.load(f)
            print(f"[vocab] carregat de {vp} (size={len(vocab)})")
        else:
            print("[vocab] construint des de HF dataset...")
            vocab = build_vocab_hf(hf_ds, threshold=args.vocab_threshold)
            # build_vocab_hf: versió de build_vocab que llegeix les captions del dataset HF
            vp.parent.mkdir(parents=True, exist_ok=True)
            with open(vp, "wb") as f:
                import pickle; pickle.dump(vocab, f)
            print(f"[vocab] built and saved to {vp} (size={len(vocab)})")

        train_loader, val_loader, _ = get_loaders_hf(
            hf_ds, vocab,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        # get_loaders_hf: versió de get_loaders que llegeix del DatasetDict de HuggingFace

        # Construim les llistes d'IDs i el DataFrame de captions
        # per a la fase SCST i per a l'avaluació BLEU final.
        full       = hf_ds["test"]
        train_rows = full.filter(lambda x: x["split"] == "train")
        val_rows   = full.filter(lambda x: x["split"] == "val")
        test_rows  = full.filter(lambda x: x["split"] == "test")
        train_ids  = [r["filename"] for r in train_rows]
        val_ids    = [r["filename"] for r in val_rows]
        test_ids   = [r["filename"] for r in test_rows]

        # Construim un DataFrame equivalent al CSV de Flickr8k per a l'avaluació BLEU
        records = []
        for r in full:
            for cap in r["caption"]:
                records.append({"image": r["filename"], "caption": cap})
        import pandas as _pd
        df_caps_hf = _pd.DataFrame(records)

        # Diccionari {nom_fitxer: PIL.Image} per a les imatges de test
        # (per generar captions durant l'avaluació BLEU sense llegir del disc)
        test_pil = {r["filename"]: r["image"] for r in test_rows}

    else:
        # ── Mode CSV local (Flickr8k per defecte) ─────────────────────────────
        vocab = get_or_build_vocab(args)
        train_loader, val_loader, _, (train_ids, val_ids, test_ids) = get_loaders(
            images_dir=args.images_dir,
            captions_csv=args.captions_csv,
            vocab=vocab,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    print(f"[vocab] size = {len(vocab)}")
    print(f"[data] train batches={len(train_loader)}  val batches={len(val_loader)}")


    # ─── 2. ENCODER ────────────────────────────────────────────────────────────
    encoder = EncoderCNNAttention(backbone=args.backbone).to(device)


    # ─── 3. EMBEDDINGS ─────────────────────────────────────────────────────────

    scst_only = (args.epochs == 0 and args.scst_checkpoint is not None)
    # scst_only = True significa que no farem la fase CE,
    # saltarem directament a SCST carregant un checkpoint prèviament entrenat.

    pretrained_weights = None
    if scst_only:
        # Si fem SCST-only, els embeddings ja estan dins el checkpoint.
        # Llegim els hiperparàmetres del checkpoint per construir l'arquitectura exacta.
        _meta = torch.load(args.scst_checkpoint, map_location="cpu")["args"]
        args.embed_size    = _meta.get("embed_size",    args.embed_size)
        args.hidden_size   = _meta.get("hidden_size",   args.hidden_size)
        args.attention_dim = _meta.get("attention_dim", args.attention_dim)
        emb_type = "from_checkpoint"
        print(f"[SCST-only] saltant càrrega de GloVe/Word2Vec — embed_size={args.embed_size}")

    elif args.glove_path:
        # GloVe té prioritat sobre Word2Vec (si s'especifiquen tots dos)
        pretrained_weights, glove_dim = load_glove_weights(args.glove_path, vocab)
        pretrained_weights = pretrained_weights.to(device)
        args.embed_size    = glove_dim  # forcem embed_size a la dim de GloVe
        emb_type = "glove-frozen" if args.freeze_embeddings else "glove-finetune"

    elif args.word2vec_path:
        binary = args.word2vec_binary if args.word2vec_binary else None
        pretrained_weights, w2v_dim = load_word2vec_weights(
            args.word2vec_path, vocab, binary=binary
        )
        pretrained_weights = pretrained_weights.to(device)
        args.embed_size    = w2v_dim
        emb_type = "word2vec-frozen" if args.freeze_embeddings else "word2vec-finetune"

    else:
        emb_type = "scratch"  # inicialització aleatòria
    print(f"[embeddings] tipus={emb_type}  embed_size={args.embed_size}")


    # ─── 4. DECODER ────────────────────────────────────────────────────────────
    decoder = AttentionDecoder(
        encoder_dim=encoder.encoder_dim,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        pretrained_weights=pretrained_weights,
        freeze_embeddings=args.freeze_embeddings,
    ).to(device)


    # ─── 5. FUNCIÓ DE PÈRDUA ───────────────────────────────────────────────────

    if pretrained_weights is not None and not args.no_semantic_loss:
        # Cas 1: embeddings preentrenats + loss semàntica
        # La SemanticCrossEntropyLoss penalitza menys paraules semànticament similars.
        # PERÒ els experiments han mostrat que distorsiona la distribució i
        # augmenta la perplexitat → resultats pitjors que CE estàndard.
        soft_lbls = build_soft_labels(
            decoder.embed.weight.data.cpu(),
            temperature=args.semantic_temp,
        )
        criterion = SemanticCrossEntropyLoss(soft_lbls).to(device)
        print(f"[loss] SemanticCrossEntropy (temp={args.semantic_temp}) — {emb_type}")

    else:
        # Cas 2: CE estàndard (amb o sense label smoothing)
        # --no-semantic-loss permet usar GloVe + CE estàndard,
        # que és la CONFIGURACIÓ GUANYADORA dels experiments actuals.
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        # label_smoothing: si > 0, en lloc de targets one-hot ([0,...,1,...,0]),
        # usa targets suavitzats: el 1 es converteix en (1-ε) i ε es distribueix
        # uniformement entre totes les altres classes.
        # Efecte: el model no s'entrena per ser "infinitament confiat", és més robust.
        # label_smoothing=0.0 = CrossEntropy estàndard (defecte).
        ls_tag = f" (label_smoothing={args.label_smoothing})" if args.label_smoothing > 0 else ""
        print(f"[loss] CrossEntropyLoss estàndard{ls_tag}")


    # ─── 6. OPTIMITZADOR I SCHEDULER ───────────────────────────────────────────
    optimizer = torch.optim.Adam(list(decoder.parameters()), lr=args.lr)
    # IMPORTANT: NOMÉS entrenem el decoder. L'encoder (ResNet) és congelat
    # (tret que s'activi --finetune-cnn-epoch, on s'afegeix layer4 als params).

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    # Redueix lr a la meitat si val_loss no millora en 2 èpoques consecutives.


    # ─── 7. WANDB ─────────────────────────────────────────────────────────────
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
        wandb.config.update({"vocab_size": len(vocab)})


    # ─── 8. VARIABLES DE CONTROL ───────────────────────────────────────────────
    train_losses:    list[float] = []
    val_losses:      list[float] = []
    best_val_loss    = float("inf")
    patience_counter = 0
    global_step      = 0
    start_epoch      = 1  # per defecte comencem des de l'época 1


    # ─── 9. REPRESA D'ENTRENAMENT (resume) ────────────────────────────────────
    if args.resume_from:
        # Si s'especifica un checkpoint, carreguem els pesos i continuem des de
        # l'época SEGÜENT a la que estava guardada.
        print(f"[resume] carregant checkpoint: {args.resume_from}")
        res = torch.load(args.resume_from, map_location=device, weights_only=False)
        encoder.load_state_dict(res["encoder"])
        decoder.load_state_dict(res["decoder"])
        start_epoch = res["epoch"] + 1  # continuem des de l'época posterior
        print(f"[resume] continuant des de l'epoch {res['epoch']} → inici epoch {start_epoch}")


    # ─── 10. BUCLE D'ENTRENAMENT CE ────────────────────────────────────────────

    for epoch in range(start_epoch, args.epochs + 1):

        # ── Fine-tuning de la CNN (si cal) ────────────────────────────────────
        if args.finetune_cnn_epoch and epoch == args.finetune_cnn_epoch:
            # A partir d'aquesta época, desbloquegem la última capa convolucional
            # de la ResNet (layer4) per fer fine-tuning.
            # encoder.cnn[-1] és la última subcapa de nn.Sequential (= layer4 de ResNet).
            encoder.finetuning = True
            for p in encoder.cnn[-1].parameters():
                p.requires_grad = True  # permet calcular gradients per a layer4

            # Afegim layer4 com a nou grup de paràmetres a l'optimitzador,
            # amb una taxa d'aprenentatge 10 vegades menor que el decoder.
            # Raó: la CNN ja té pesos bons (ImageNet); no volem canviar-los massa.
            optimizer.add_param_group({
                "params": list(encoder.cnn[-1].parameters()),
                "lr": args.lr / 10,
            })
            print(f"[finetune] epoch {epoch}: layer4 descongelada (lr={args.lr/10:.2e})")

        encoder.train()  # activa mode train (batch norm en mode train)
        decoder.train()  # activa dropout
        t0 = time.time()

        for i, (images, captions, lengths) in enumerate(train_loader):
            images   = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            # Preparem els targets: la caption sense <start>, compactada
            targets = pack_padded_sequence(
                captions[:, 1:],           # [B, T-1] — sense el primer token (<start>)
                [l - 1 for l in lengths],  # longituds ajustades
                batch_first=True,
            ).data

            features = encoder(images)
            # [B, 3, 224, 224] → [B, 49, 2048]

            outputs, alphas_sum = decoder(features, captions, lengths)
            # outputs:    [sum(lengths-1), vocab_size] — prediccions dels tokens
            # alphas_sum: [B] — suma dels pesos d'atenció per a cada imatge
            #             Usada per a la regularització Doubly Stochastic.

            loss = criterion(outputs, targets)
            # Cross-Entropy (o Semantic CE) entre prediccions i tokens correctes

            # ── Doubly Stochastic Attention regularitzation ────────────────────
            if args.ds_lambda > 0:
                loss = loss + args.ds_lambda * ((1 - alphas_sum) ** 2).mean()
            # Concepte Doubly Stochastic (Xu et al., 2015):
            #   Si el model atén correctament, la suma dels pesos d'atenció per a
            #   cada pas de la generació hauria de ser aproximadament 1 per a cada region.
            #   És a dir, al llarg de tots els passos, el model hauria de "veure"
            #   totes les regions de la imatge aproximadament una vegada.
            #   (1 - alphas_sum)^2 penalitza si alphas_sum s'allunya de 1.
            #   ds_lambda=1.0: la regularització pesa igual que la loss principal.
            #   ds_lambda=0:   regularització desactivada.

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            # Gradient clipping: evita que gradients molt grans desestabilitzin l'entrenament.
            # Si la norma L2 de tots els gradients supera 5.0, els escala tots proporcionalment.
            # Especialment útil amb LSTMs que poden tenir gradients explosius.

            optimizer.step()

            global_step += 1
            train_losses.append(loss.item())

            if i % args.log_step == 0:
                ppl = float(np.exp(min(loss.item(), 20)))
                print(f"epoch {epoch}/{args.epochs}  step {i}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  ppl={ppl:.2f}")
                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(), "train/perplexity": ppl,
                        "epoch": epoch, "step": global_step,
                    })

        # ── Avaluació de validació ──────────────────────────────────────────────
        val_loss = evaluate(encoder, decoder, val_loader, criterion, device)
        val_losses.append(val_loss)
        scheduler.step(val_loss)  # possiblement redueix lr si no hi ha millora
        val_ppl = float(np.exp(min(val_loss, 20)))
        elapsed = time.time() - t0
        print(f"== epoch {epoch} done  val_loss={val_loss:.4f}  "
              f"val_ppl={val_ppl:.2f}  ({elapsed:.0f}s)")
        if use_wandb:
            wandb.log({
                "val/loss": val_loss, "val/perplexity": val_ppl,
                "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            })

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt = {
            "epoch":      epoch,
            "encoder":    encoder.state_dict(),
            "decoder":    decoder.state_dict(),
            "vocab_size": len(vocab),
            "args":       vars(args),
        }
        out = Path(args.checkpoints_dir) / f"ckpt_epoch{epoch}.pt"
        safe_save(ckpt, out)  # guarda amb reintents per errors NFS
        print(f"[ckpt] saved {out}")

        # ── Early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            safe_save(ckpt, Path(args.checkpoints_dir) / "ckpt_best.pt")
            print(f"[early_stop] new best val_loss={best_val_loss:.4f} → saved ckpt_best.pt")
        else:
            patience_counter += 1
            print(f"[early_stop] no improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"[early_stop] patience exhausted, stopping at epoch {epoch}")
                break


    # ─── 11. GRÀFIC DE PÈRDUA ──────────────────────────────────────────────────
    if train_losses:
        # Només generem el gràfic si s'ha fet almenys una época d'entrenament CE
        steps_per_epoch = len(train_loader)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(train_losses, alpha=0.6, label="train (per batch)")
        for e in range(1, args.epochs + 1):
            axes[0].axvline(e * steps_per_epoch, color="gray", linestyle="--", linewidth=0.8)
        axes[0].set_xlabel("batch")
        axes[0].set_ylabel("cross-entropy loss")
        axes[0].set_title("Train loss (attention)")
        axes[0].legend()
        axes[1].plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="val")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("cross-entropy loss")
        axes[1].set_title("Val loss per epoch (attention)")
        axes[1].legend()
        plt.tight_layout()
        plot_path = Path(args.checkpoints_dir) / "loss_curve.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[plot] saved {plot_path}")


    # ─── 12. FASE SCST (fine-tuning amb CIDEr-D reward) ───────────────────────

    best_ckpt_path = Path(args.checkpoints_dir) / "ckpt_best.pt"
    # Punt de partida per a l'avaluació BLEU final.
    # S'actualitza si es fa la fase SCST.

    if args.scst_epochs > 0:
        print("\n[SCST] Iniciant fine-tuning SCST amb CIDEr-D reward...")
        Path(args.scst_checkpoints_dir).mkdir(parents=True, exist_ok=True)

        # Checkpoint inicial per a SCST: el que s'ha especificat o el millor del CE
        scst_start = args.scst_checkpoint or str(best_ckpt_path)
        if not Path(scst_start).exists():
            raise FileNotFoundError(
                f"[SCST] Checkpoint not found: {scst_start}\n"
                "Run CE training first, or pass --scst-checkpoint <path>."
            )
        print(f"[SCST] Carregant checkpoint: {scst_start}")
        scst_ckpt = torch.load(scst_start, map_location=device)
        encoder.load_state_dict(scst_ckpt["encoder"])
        decoder.load_state_dict(scst_ckpt["decoder"])

        # Congelar totalment l'encoder durant SCST per estabilitat
        encoder.finetuning = False
        for p in encoder.parameters():
            p.requires_grad = False

        # Pre-calcula IDF una sola vegada (O(|train|))
        if args.flickr30k_hf:
            df_train_caps = df_caps_hf[df_caps_hf["image"].isin(set(train_ids))].reset_index(drop=True)
        else:
            df_all        = load_captions_df(args.captions_csv)
            df_train_caps = df_all[df_all["image"].isin(set(train_ids))].reset_index(drop=True)
        idf = compute_idf_weights(df_train_caps)

        # Diccionari {img_id: [llista de captions]} per a calcular CIDEr-D durant SCST
        refs_by_image: dict = {}
        for img_id in train_ids:
            refs_by_image[img_id] = df_train_caps[
                df_train_caps["image"] == img_id
            ]["caption"].tolist()

        # Optimitzador separat per a SCST amb lr molt baix
        scst_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.scst_lr)

        # DataLoader especial per a SCST (inclou img_ids al batch)
        if args.flickr30k_hf:
            scst_loader = get_scst_loader_hf(
                hf_ds, vocab,
                batch_size=args.scst_batch_size,
                num_workers=args.num_workers,
            )
        else:
            scst_loader = get_scst_loader(
                images_dir=args.images_dir,
                captions_csv=args.captions_csv,
                vocab=vocab,
                train_ids=train_ids,
                batch_size=args.scst_batch_size,
                num_workers=args.num_workers,
            )
        print(f"[SCST] {len(scst_loader)} batches/epoch  "
              f"lr={args.scst_lr}  batch_size={args.scst_batch_size}")

        best_scst_reward = float("-inf")
        for scst_ep in range(1, args.scst_epochs + 1):
            t0 = time.time()
            mean_reward = scst_epoch(
                encoder, decoder, scst_loader, scst_optimizer,
                vocab, refs_by_image, idf, device, args,
                use_wandb=use_wandb, epoch=scst_ep,
            )
            elapsed = time.time() - t0
            print(f"== SCST epoch {scst_ep}/{args.scst_epochs}  "
                  f"mean_reward={mean_reward:.4f}  ({elapsed:.0f}s)")
            if use_wandb:
                wandb.log({"scst/epoch_reward": mean_reward, "scst/epoch": scst_ep})

            scst_save = {
                "epoch":        scst_ep,
                "encoder":      encoder.state_dict(),
                "decoder":      decoder.state_dict(),
                "vocab_size":   len(vocab),
                "args":         vars(args),
                "scst_reward":  mean_reward,
            }
            out = Path(args.scst_checkpoints_dir) / f"ckpt_scst_epoch{scst_ep}.pt"
            safe_save(scst_save, out)
            print(f"[SCST ckpt] saved {out}")

            if mean_reward > best_scst_reward:
                best_scst_reward = mean_reward
                safe_save(scst_save, Path(args.scst_checkpoints_dir) / "ckpt_best.pt")
                print(f"[SCST] new best reward={best_scst_reward:.4f} → saved ckpt_best.pt")

        # Actualitzem el path del millor checkpoint per a l'avaluació BLEU
        best_ckpt_path = Path(args.scst_checkpoints_dir) / "ckpt_best.pt"
        print(f"[SCST] Fine-tuning done. BLEU eval will use {best_ckpt_path}")


    # ─── 13. AVALUACIÓ FINAL: BLEU + METEOR ────────────────────────────────────
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    print("\n[bleu+meteor] avaluant sobre el conjunt de test...")

    # Carreguem el millor model (CE o SCST)
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    encoder.load_state_dict(best_ckpt["encoder"])
    decoder.load_state_dict(best_ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    if args.flickr30k_hf:
        df_caps = df_caps_hf
    else:
        df_caps = load_captions_df(args.captions_csv)
    smooth = SmoothingFunction().method1

    all_refs, all_hyps, all_meteors = [], [], []
    bleu_table = (
        wandb.Table(columns=["image", "generated_caption", "reference_captions", "BLEU-1", "BLEU-4", "METEOR"])
        if use_wandb else None
    )
    images_dir_abs = Path(args.images_dir).resolve()
    TABLE_LIMIT    = 200  # màxim de files a la taula WandB (per rendiment)

    print(f"Evaluating {len(test_ids)} test images...")
    print(f"{'Image':<35} {'BLEU-1':>7} {'BLEU-4':>7} {'METEOR':>7}  Caption")
    print("-" * 110)

    for img in test_ids:
        # Referencies: les 5 captions humanes de la imatge
        refs = [
            simple_tokenize(c)
            for c in df_caps[df_caps["image"] == img]["caption"].tolist()
        ]

        # Hipòtesi: caption generada pel model (amb beam search)
        if args.flickr30k_hf:
            hyp = simple_tokenize(caption_pil_image(test_pil[img], encoder, decoder, vocab, device))
            # Usem la imatge PIL directament (HuggingFace)
        else:
            hyp = simple_tokenize(caption_image(f"{args.images_dir}/{img}", encoder, decoder, vocab, device))
            # Llegim la imatge des del disc (Flickr8k local)

        # Mètriques per a aquesta imatge
        b1 = sentence_bleu(refs, hyp, weights=(1, 0, 0, 0),         smoothing_function=smooth)
        b4 = sentence_bleu(refs, hyp, weights=(.25, .25, .25, .25), smoothing_function=smooth)
        m  = meteor_score(refs, hyp)

        all_refs.append(refs)
        all_hyps.append(hyp)
        all_meteors.append(m)
        print(f"{img:<35} {b1:>7.3f} {b4:>7.3f} {m:>7.3f}  {' '.join(hyp)}")

        # Afegim la fila a la taula WandB (si s'usa)
        if bleu_table is not None and len(bleu_table.data) < TABLE_LIMIT:
            ref_str = " | ".join([" ".join(r) for r in refs])
            if args.flickr30k_hf:
                bleu_table.add_data(
                    str(img), " ".join(hyp), ref_str,
                    round(b1, 3), round(b4, 3), round(m, 3),
                )
            else:
                bleu_table.add_data(
                    wandb.Image(str(images_dir_abs / img)), " ".join(hyp), ref_str,
                    round(b1, 3), round(b4, 3), round(m, 3),
                )

    # Mètriques de corpus (sobre totes les imatges de test alhora)
    cb1 = corpus_bleu(all_refs, all_hyps, weights=(1, 0, 0, 0))
    cb4 = corpus_bleu(all_refs, all_hyps, weights=(.25, .25, .25, .25))
    cm  = float(np.mean(all_meteors))
    print("-" * 110)
    print(f"[bleu] Corpus BLEU-1: {cb1:.3f}  BLEU-4: {cb4:.3f}  METEOR: {cm:.3f}")

    if use_wandb:
        wandb.log({
            "bleu/corpus_bleu1": cb1,
            "bleu/corpus_bleu4": cb4,
            "bleu/meteor":       cm,
            "bleu/eval_table":   bleu_table,
        })
        wandb.finish()


if __name__ == "__main__":
    main()