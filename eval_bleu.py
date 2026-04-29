"""BLEU evaluation on the full test set, logged to wandb."""
# Docstring: aquest script avalua la qualitat de les descripcions generades
# usant la mètrica BLEU sobre el conjunt de test complet, i ho registra a wandb.

import pandas as pd          # per llegir el CSV de captions
import torch                 # framework de deep learning
import wandb                 # per registrar mètriques i resultats a la web
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
# corpus_bleu  → BLEU global sobre tot el conjunt
# sentence_bleu → BLEU per a una sola imatge
# SmoothingFunction → evita puntuació 0 quan no hi ha n-grams coincidents

from src.dataset import split_image_ids   # divideix IDs en train/val/test
from src.sample import caption_image, load_checkpoint  # genera captions i carrega el model
from src.vocabulary import simple_tokenize  # tokenitza text en llista de paraules

# ── Paths ──────────────────────────────────────────────────────────────────
CHECKPOINT   = "checkpoints/ckpt_best.pt"          # model guardat (el millor)
VOCAB_PATH   = "data/flickr8k/vocab.pkl"            # vocabulari serialitzat
IMAGES_DIR   = "data/flickr8k/Images"              # carpeta d'imatges
CAPTIONS_CSV = "data/flickr8k/captions.txt"        # fitxer amb totes les captions

# ── Dispositiu i model ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# usa GPU si n'hi ha, sinó CPU

encoder, decoder, vocab = load_checkpoint(CHECKPOINT, VOCAB_PATH, device)
# carrega l'encoder (CNN), el decoder (RNN/Transformer) i el vocabulari

# ── Dades de test ───────────────────────────────────────────────────────────
_, _, test_ids = split_image_ids(CAPTIONS_CSV)
# obté només els IDs del conjunt de test (descarta train i val)

df = pd.read_csv(CAPTIONS_CSV)
# carrega tot el CSV amb columnes "image" i "caption"

smooth = SmoothingFunction().method1
# funció de suavitzat method1: assigna un petit valor als n-grams no trobats
# (necessari per a frases curtes que donarien BLEU-4 = 0)

# ── WandB ───────────────────────────────────────────────────────────────────
run = wandb.init(entity="learning6", project="image-captioning",
                 name="bleu-eval-fulltest", config={"n_images": len(test_ids), "checkpoint": CHECKPOINT})
# inicia una sessió de wandb al projecte "image-captioning"
# guarda com a metadades: nombre d'imatges avaluades i checkpoint usat

# ── Llistes acumuladores ─────────────────────────────────────────────────────
all_refs, all_hyps = [], []
# all_refs → totes les referències (captions humanes), per al corpus_bleu final
# all_hyps → totes les hipòtesis (captions generades), per al corpus_bleu final

table = wandb.Table(columns=["image", "generated_caption", "reference_captions", "BLEU-1", "BLEU-4"])
# taula wandb per visualitzar cada imatge amb la seva caption i puntuació

# ── Capçalera de la consola ──────────────────────────────────────────────────
print(f"Evaluating {len(test_ids)} test images...")
print(f"{'Image':<35} {'BLEU-1':>7} {'BLEU-4':>7}  Caption")
print("-" * 100)
# imprimeix una capçalera formatada per veure els resultats per imatge

# ── Bucle principal d'avaluació ──────────────────────────────────────────────
for img in test_ids:
    # itera sobre cada ID d'imatge del conjunt de test

    refs = [simple_tokenize(c) for c in df[df["image"] == img]["caption"].tolist()]
    # filtra el DataFrame per obtenir les captions d'aquesta imatge (normalment 5)
    # i les tokenitza → llista de llistes de paraules  ex: [["a","dog",...], ...]

    hyp  = simple_tokenize(caption_image(f"{IMAGES_DIR}/{img}", encoder, decoder, vocab, device))
    # genera una caption automàtica per a la imatge i la tokenitza
    # hyp és la "hipòtesi" que volem avaluar  ex: ["a","dog","playing","in","water"]

    b1 = sentence_bleu(refs, hyp, weights=(1,0,0,0), smoothing_function=smooth)
    # BLEU-1: només compta unigrames coincidents (precisió de paraules individuals)

    b4 = sentence_bleu(refs, hyp, weights=(.25,.25,.25,.25), smoothing_function=smooth)
    # BLEU-4: mitjana geomètrica de 1-grams, 2-grams, 3-grams i 4-grams
    # és la mètrica estàndard per avaluar generació de text

    all_refs.append(refs)   # acumula per al BLEU de corpus al final
    all_hyps.append(hyp)    # acumula per al BLEU de corpus al final

    ref_str = " | ".join([" ".join(r) for r in refs])
    # ajunta totes les captions de referència en un sol string separat per "|"
    # per mostrar-les a la taula de wandb

    table.add_data(wandb.Image(f"{IMAGES_DIR}/{img}"), " ".join(hyp), ref_str, round(b1, 3), round(b4, 3))
    # afegeix una fila a la taula wandb: imatge, caption generada, referències, BLEU-1, BLEU-4

    print(f"{img:<35} {b1:>7.3f} {b4:>7.3f}  {' '.join(hyp)}")
    # imprimeix a consola: nom imatge, BLEU-1, BLEU-4 i la caption generada

# ── BLEU de corpus (global) ──────────────────────────────────────────────────
cb1 = corpus_bleu(all_refs, all_hyps, weights=(1,0,0,0))
cb4 = corpus_bleu(all_refs, all_hyps, weights=(.25,.25,.25,.25))
# calcula el BLEU sobre TOT el conjunt de test a la vegada
# és més fiable que la mitjana dels BLEU individuals perquè
# la penalització de brevetat (BP) s'aplica globalment

print("-" * 100)
print(f"{'Corpus BLEU':<35} {cb1:>7.3f} {cb4:>7.3f}")
# imprimeix els resultats finals de corpus BLEU

# ── Registre final a wandb ───────────────────────────────────────────────────
wandb.log({"bleu/corpus_bleu1": cb1, "bleu/corpus_bleu4": cb4, "eval_table": table})
# puja a wandb: les dues mètriques globals i la taula completa d'imatges

print(f"\nWandb: {run.url}")   # mostra l'enllaç a la sessió de wandb
wandb.finish()                 # tanca la sessió correctament
