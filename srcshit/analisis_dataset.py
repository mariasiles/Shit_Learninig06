"""Anàlisi completa del dataset Flickr8k."""
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT      = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT.parent  # data/ i checkpoints/ estan al projecte principal
sys.path.insert(0, str(ROOT))

from src.vocabulary import build_vocab, simple_tokenize
from src.dataset import split_image_ids, get_loaders

CAPTIONS_CSV = str(DATA_ROOT / "data/flickr8k/captions.txt")
IMAGES_DIR   = str(DATA_ROOT / "data/flickr8k/Images")
VOCAB_PATH   = str(DATA_ROOT / "data/flickr8k/vocab.pkl")
OUT_DIR      = DATA_ROOT / "checkpoints/dataset_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

log_lines = []

def log(text=""):
    print(text)
    log_lines.append(text)

df = pd.read_csv(CAPTIONS_CSV)

# ── 1. Mides generals ────────────────────────────────────────────────────────
n_images   = df["image"].nunique()
n_captions = len(df)
log("=" * 60)
log("1. MIDES GENERALS")
log(f"   Imatges úniques : {n_images}")
log(f"   Captions totals : {n_captions}")
log(f"   Captions/imatge : {n_captions/n_images:.1f}")

# ── 2. Train / Val / Test split ───────────────────────────────────────────────
train_ids, val_ids, test_ids = split_image_ids(CAPTIONS_CSV)
log("\n2. SPLIT")
log(f"   Train : {len(train_ids):>5} imatges  ({len(train_ids)/n_images*100:.1f}%)  → {len(train_ids)*5} captions")
log(f"   Val   : {len(val_ids):>5} imatges  ({len(val_ids)/n_images*100:.1f}%)  → {len(val_ids)*5} captions")
log(f"   Test  : {len(test_ids):>5} imatges  ({len(test_ids)/n_images*100:.1f}%)  → {len(test_ids)*5} captions")

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie([len(train_ids), len(val_ids), len(test_ids)],
       labels=["Train", "Val", "Test"], autopct="%1.1f%%",
       colors=["#4C72B0", "#DD8452", "#55A868"])
ax.set_title("Distribució del dataset")
fig.savefig(OUT_DIR / "split_pie.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 3. Estadístiques de les captions ─────────────────────────────────────────
df["tokens"] = df["caption"].astype(str).apply(simple_tokenize)
df["length"] = df["tokens"].apply(len)
log("\n3. LONGITUD DE LES CAPTIONS (paraules)")
log(df["length"].describe().round(2).to_string())

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df["length"], bins=30, color="#4C72B0", edgecolor="white")
ax.axvline(df["length"].mean(), color="red", linestyle="--",
           label=f"Mitjana: {df['length'].mean():.1f}")
ax.set_xlabel("Nombre de paraules")
ax.set_ylabel("Freqüència")
ax.set_title("Distribució de la longitud de les captions")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "caption_lengths.png", dpi=150)
plt.close()

# ── 4. Vocabulari ─────────────────────────────────────────────────────────────
vocab   = build_vocab(CAPTIONS_CSV, threshold=5)
counter = Counter(t for tokens in df["tokens"] for t in tokens)
coverage = sum(c for w, c in counter.items() if c >= 5) / sum(counter.values())
log("\n4. VOCABULARI")
log(f"   Paraules úniques totals : {len(counter)}")
log(f"   Vocabulari (thresh≥5)   : {len(vocab)}")
log(f"   Cobertura               : {coverage*100:.1f}%")
log(f"   Top 10: {[w for w, _ in counter.most_common(10)]}")

words, counts = zip(*counter.most_common(20))
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(words, counts, color="#4C72B0")
ax.set_title("Top 20 paraules més freqüents")
ax.set_ylabel("Freqüència")
plt.tight_layout()
fig.savefig(OUT_DIR / "top_words.png", dpi=150)
plt.close()

# ── 5. Mides de les imatges ───────────────────────────────────────────────────
sample_imgs = random.sample(list(df["image"].unique()), 200)
widths, heights = [], []
for img in sample_imgs:
    with Image.open(f"{IMAGES_DIR}/{img}") as im:
        w, h = im.size
        widths.append(w)
        heights.append(h)

log("\n5. MIDES DE LES IMATGES (mostra 200)")
log(f"   Amplada — mitjana: {np.mean(widths):.0f}px  min: {min(widths)}  max: {max(widths)}")
log(f"   Alçada  — mitjana: {np.mean(heights):.0f}px  min: {min(heights)}  max: {max(heights)}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(widths, bins=20, color="#4C72B0", edgecolor="white")
axes[0].set_title("Distribució amplada (px)")
axes[1].hist(heights, bins=20, color="#DD8452", edgecolor="white")
axes[1].set_title("Distribució alçada (px)")
plt.tight_layout()
fig.savefig(OUT_DIR / "image_sizes.png", dpi=150)
plt.close()

# ── 6. DataLoader ─────────────────────────────────────────────────────────────
with open(VOCAB_PATH, "rb") as f:
    vocab_loaded = pickle.load(f)

train_loader, val_loader, test_loader, _ = get_loaders(
    images_dir=IMAGES_DIR, captions_csv=CAPTIONS_CSV,
    vocab=vocab_loaded, batch_size=32, num_workers=2,
)
images, captions, lengths = next(iter(train_loader))
log("\n6. DATALOADER (batch_size=32)")
log(f"   Train batches : {len(train_loader)}")
log(f"   Val   batches : {len(val_loader)}")
log(f"   Test  batches : {len(test_loader)}")
log(f"   Shape imatges : {tuple(images.shape)}  (B x C x H x W)")
log(f"   Shape captions: {tuple(captions.shape)}  (B x T_max)")
log(f"   Imatges normalitzades — min: {images.min():.3f}  max: {images.max():.3f}")

log(f"\nGràfiques guardades a {OUT_DIR}/")

with open(OUT_DIR / "stats.txt", "w") as f:
    f.write("\n".join(log_lines))
log(f"Stats guardades a {OUT_DIR}/stats.txt")
