"""
Anàlisi del DataLoader - Flickr8k
----------------------------------
Aquest script analitza el dataset per entendre:
  - Quantes imatges i peus de foto tenim
  - Com es divideix en train/val/test
  - Si les dades estan normalitzades
  - Estadístiques bàsiques dels peus de foto (longitud, paraules freqüents...)
"""

import sys
from pathlib import Path

# Afegim el directori arrel al path perquè trobi els mòduls locals
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

try:
    from src.vocabulary import build_vocab, simple_tokenize
    from src.dataset import get_loaders, split_image_ids
except ImportError:
    from vocabulary import build_vocab, simple_tokenize
    from dataset import get_loaders, split_image_ids

# ---- CONFIGURACIÓ ----
# Modifica aquestes rutes si el teu dataset és en un altre lloc
IMAGES_DIR = "dataset/Images"
CAPTIONS_CSV = "dataset/captions.txt"


# =====================================================================
# 1. LECTURA DEL CSV
# =====================================================================
print("=" * 55)
print("1. INFORMACIÓ BÀSICA DEL DATASET")
print("=" * 55)

df = pd.read_csv(CAPTIONS_CSV)

# Quantes files té el CSV? (una per cada peu de foto)
print(f"Total de files al CSV:       {len(df)}")

# Quantes imatges ÚNIQUES tenim?
num_imatges = df["image"].nunique()
print(f"Imatges úniques:             {num_imatges}")

# Cada imatge té quants peus de foto de mitja?
peus_per_imatge = len(df) / num_imatges
print(f"Peus de foto per imatge:     {peus_per_imatge:.1f} (de mitja)")


# =====================================================================
# 2. DIVISIÓ TRAIN / VAL / TEST
# =====================================================================
print("\n" + "=" * 55)
print("2. DIVISIÓ TRAIN / VAL / TEST")
print("=" * 55)

train_ids, val_ids, test_ids = split_image_ids(CAPTIONS_CSV)

print(f"Train: {len(train_ids)} imatges  ({len(train_ids)/num_imatges*100:.1f}%)")
print(f"Val:   {len(val_ids)} imatges   ({len(val_ids)/num_imatges*100:.1f}%)")
print(f"Test:  {len(test_ids)} imatges   ({len(test_ids)/num_imatges*100:.1f}%)")

# Nombre de mostres per split (imatges x peus de foto)
print(f"\nMostres per al DataLoader:")
print(f"  Train: {len(train_ids) * 5} (cada imatge té 5 peus de foto)")
print(f"  Val:   {len(val_ids) * 5}")
print(f"  Test:  {len(test_ids) * 5}")


# =====================================================================
# 3. NORMALITZACIÓ
# =====================================================================
print("\n" + "=" * 55)
print("3. NORMALITZACIÓ DE LES IMATGES")
print("=" * 55)

# El dataset.py utilitza la normalització estàndard d'ImageNet
# perquè el encoder (ResNet-50) ha estat pre-entrenat amb ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

print("Sí, les imatges estan normalitzades amb els valors d'ImageNet:")
print(f"  Mean (R, G, B): {IMAGENET_MEAN}")
print(f"  Std  (R, G, B): {IMAGENET_STD}")
print("Això és important perquè la ResNet-50 ha après amb aquestes escales.")


# =====================================================================
# 4. ESTADÍSTIQUES DELS PEUS DE FOTO
# =====================================================================
print("\n" + "=" * 55)
print("4. ESTADÍSTIQUES DELS PEUS DE FOTO")
print("=" * 55)

# Calculem la longitud (en paraules) de cada peu de foto
df["longitud"] = df["caption"].astype(str).apply(lambda x: len(simple_tokenize(x)))

print(f"Longitud mínima:   {df['longitud'].min()} paraules")
print(f"Longitud màxima:   {df['longitud'].max()} paraules")
print(f"Longitud mitjana:  {df['longitud'].mean():.1f} paraules")
print(f"Longitud mediana:  {df['longitud'].median():.1f} paraules")


# =====================================================================
# 5. VOCABULARI
# =====================================================================
print("\n" + "=" * 55)
print("5. VOCABULARI")
print("=" * 55)

# Construïm el vocabulari (paraules que apareixen >= 5 vegades)
vocab = build_vocab(CAPTIONS_CSV, threshold=5)
print(f"Paraules al vocabulari (threshold=5): {len(vocab)}")

# Quines paraules apareixen més?
all_words = []
for cap in df["caption"].astype(str):
    all_words.extend(simple_tokenize(cap))

paraules_freq = Counter(all_words).most_common(10)
print("\nTop 10 paraules més freqüents:")
for paraula, freq in paraules_freq:
    print(f"  '{paraula}': {freq} vegades")


# =====================================================================
# 6. GRÀFIQUES
# =====================================================================
print("\n" + "=" * 55)
print("6. GRÀFIQUES (es guarden com a .png)")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Anàlisi del Dataset Flickr8k", fontsize=14)

# Gràfica 1: Distribució de longituds dels peus de foto
axes[0].hist(df["longitud"], bins=20, color="steelblue", edgecolor="white")
axes[0].set_title("Distribució de longituds dels captions")
axes[0].set_xlabel("Nombre de paraules")
axes[0].set_ylabel("Freqüència")
axes[0].axvline(df["longitud"].mean(), color="red", linestyle="--", label=f"Mitjana: {df['longitud'].mean():.1f}")
axes[0].legend()

# Gràfica 2: Divisió del dataset
splits = ["Train", "Val", "Test"]
mides = [len(train_ids), len(val_ids), len(test_ids)]
axes[1].bar(splits, mides, color=["steelblue", "orange", "green"])
axes[1].set_title("Divisió del dataset (nº imatges)")
axes[1].set_ylabel("Nombre d'imatges")
for i, v in enumerate(mides):
    axes[1].text(i, v + 20, str(v), ha="center", fontweight="bold")

plt.tight_layout()
output_path = root_dir / "analisis_dataloader.png"
plt.savefig(output_path, dpi=150)
print(f"Gràfica guardada a: {output_path}")
plt.show()

print("\n✓ Anàlisi completat!")
