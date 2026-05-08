"""
dataset.py
==========
Càrrega i preparació de les dades per a l'entrenament.

Aquest fitxer respon a la pregunta: "Com passem d'imatges i textos en disc a tensors numèrics que el model pot consumir?"

Responsabilitats principals:
  1. Llegir el CSV de captions (Flickr8k o Flickr30k)
  2. Definir les transformacions que s'apliquen a cada imatge
  3. Implementar la classe Dataset (que representa la col·lecció de dades)
  4. Implementar collate_fn (que agrupa mostres en batches uniformes)
  5. Dividir les imatges en train / val / test
  6. Crear els DataLoaders finals llestos per a l'entrenament

Estructura esperada al disc:
    Flickr8k:
        Images/          ← totes les .jpg
        captions.txt     ← CSV: image,caption

    Flickr30k:
        flickr30k_images/ ← totes les .jpg
        results.csv       ← CSV: image_name|comment_number|comment

    Flickr30k (HuggingFace):
        load_dataset('nlphuji/flickr30k', cache_dir=...) — imatges ja incloses
"""

from __future__ import annotations  # permet type hints moderns (list[str], etc.)

from pathlib import Path             # rutes de fitxer multiplataforma

import pandas as pd                  # lectura i manipulació de CSV com a taules
import torch                         # tensors i operacions de xarxes neuronals
from PIL import Image                # obrir imatges .jpg des del disc
from torch.utils.data import DataLoader, Dataset
# Dataset:    classe abstracta de PyTorch per representar una col·lecció de dades.
#             Cal implementar __len__ i __getitem__.
# DataLoader: embolcall al voltant d'un Dataset que genera batches automàticament, barreja les dades, paral·lelitza la càrrega, etc.

from torchvision import transforms   # transformacions d'imatge predefinides (resize, crop, etc.)

from src.shared.vocabulary import Vocabulary  # el vocabulari que hem construït


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA DEL CSV DE CAPTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_captions_df(captions_csv: str | Path) -> pd.DataFrame:
    """
    Llegeix el CSV de captions i sempre retorna un DataFrame amb dues columnes:
       'image' (nom del fitxer) i 'caption' (text de la descripció).

    Detecta automàticament si el fitxer és Flickr8k o Flickr30k llegint
    la primera línia per veure quin separador utilitza.

    Flickr8k  usa comes   (,) i ja té columnes 'image' i 'caption'.
    Flickr30k usa barres  (|) i té 'image_name', 'comment_number', 'comment'.
    Aquí les renombrem per tenir sempre la mateixa interfície.
    """
    captions_csv = Path(captions_csv)

    # Llegim NOMÉS la capçalera per detectar el format sense carregar tot el fitxer
    with open(captions_csv, encoding="utf-8") as f:
        header = f.readline()  # primera línia: "image,caption" o "image_name| comment_number| comment"

    if "|" in header:

        # ── Flickr30k ──────────────────────────────────────────────────────
        df = pd.read_csv(captions_csv, sep="|", skipinitialspace=True)
        # skipinitialspace=True elimina els espais que hi ha just després del separador |

        df.columns = [c.strip() for c in df.columns]
        # Eliminem espais dels noms de columna: " comment" → "comment"

        df = df.rename(columns={"image_name": "image", "comment": "caption"})
        # Renombrem perquè les columnes es diguin igual que a Flickr8k

        df = df[["image", "caption"]].copy()  # ens quedem només les 2 columnes que ens interessen

        df["image"] = df["image"].str.strip()           # elimina espais del nom de fitxer
        df["caption"] = df["caption"].astype(str).str.strip()  # elimina espais de la caption

    else:
        # ── Flickr8k ───────────────────────────────────────────────────────
        df = pd.read_csv(captions_csv)
        df = df[["image", "caption"]].copy()  # ja té les columnes correctes

    return df.reset_index(drop=True)   # reset_index: torna a numerar les files des de 0 (per si s'han eliminat files)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALITZACIÓ D'IMAGENET
# ─────────────────────────────────────────────────────────────────────────────

# Per què usem els valors d'ImageNet?
# La ResNet que fem servir com a encoder va ser PREENTRENADA amb imatges d'ImageNet.
# Durant aquest preentrenament, les imatges es van normalitzar amb aquests valors.
# Si nosaltres li passem imatges sense normalitzar (o amb uns altres valors),
# la ResNet "veurà" distribucions de píxels completament diferents i no funcionarà bé.
# Per tant, hem de preparar les imatges EXACTAMENT igual que es van preparar a ImageNet.

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # mitjana per canal: vermell, verd, blau
IMAGENET_STD  = (0.229, 0.224, 0.225)  # desviació estàndard per canal: vermell, verd, blau

# La normalització fa, per a cada píxel de cada canal:
#     píxel_normalitzat = (píxel_original - mean) / std
# Resultat: els valors ja no van de 0 a 1, sinó aproximadament de -2 a +2,
# centrats al voltant de 0. Això facilita l'entrenament numèricament.


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMACIONS D'IMATGE
# ─────────────────────────────────────────────────────────────────────────────

def get_transform(image_size: int = 224, train: bool = True):
    """
    Retorna la cadena de transformacions que s'aplica a cada imatge.

    Les transformacions converteixen una imatge PIL (H x W píxels, valors 0-255)
    en un tensor PyTorch [3, 224, 224] amb valors normalitzats.

    Hi ha dues versions:
        train=True:  inclou transformacions ALEATÒRIES per a data augmentation
        train=False: transformacions DETERMINISTES per validació/test

    Per què data augmentation?
        El model veu "versions" lleugerament diferents de cada imatge en cada
        època d'entrenament. Això el força a aprendre característiques generals
        en lloc de memoritzar les imatges, millorant la generalització.

    Args:
        image_size: costat del quadrat final en píxels (per defecte 224, que és
                    el que espera la ResNet)
        train:      True per a entrenament, False per a validació/test
    """
    if train:
        return transforms.Compose([
            # 1- Redimensiona: el costat curt de la imatge passa a tenir 256 píxels,
            #    mantenint les proporcions originals.
            #    Ex: imatge 1200x800 → 384x256  |  imatge 640x480 → 341x256
            transforms.Resize(256),

            # 2- Retall aleatori: tria una posició aleatòria i retalla un quadrat
            #    de 224x224 píxels. En cada època, el model veurà un tros diferent
            #    de la mateixa imatge → l'obliga a aprendre de tot l'enquadre.
            transforms.RandomCrop(image_size),

            # 3- Flip horitzontal aleatori (probabilitat 50%): gira la imatge com un mirall.
            #    Un gos mirant a la dreta pot aparèixer mirant a l'esquerra.
            #    La caption és la mateixa, però la imatge és diferent → més variabilitat.
            transforms.RandomHorizontalFlip(),

            # 4- Converteix la imatge PIL (valors enters 0-255) a tensor PyTorch
            #    de forma [3, H, W] amb valors flotants de 0.0 a 1.0.
            transforms.ToTensor(),

            # 5- Normalitza cada canal (R, G, B) amb la mitjana i desviació d'ImageNet.
            #    Resultat: tensor [3, 224, 224] amb valors aprox. entre -2.5 i +2.5.
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        # Per a validació i test, SENSE aleatorietat:
        return transforms.Compose([
            transforms.Resize(256),           # igual que a train

            # CenterCrop en lloc de RandomCrop: sempre retalla el centre.
            # Si evaluem el mateix model dues vegades sobre les mateixes imatges, obtenim exactament el mateix resultat.
            transforms.CenterCrop(image_size),

            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE DATASET
# ─────────────────────────────────────────────────────────────────────────────

class Flickr8kDataset(Dataset):
    """Representa el dataset de parells (imatge, caption) com una col·lecció indexable.

    Hereda de torch.utils.data.Dataset, que exigeix implementar:
        __len__:     quantes mostres hi ha en total
        __getitem__: donada una posició, retorna la mostra corresponent

    Una "mostra" és un parell:
        image_tensor:   FloatTensor [3, 224, 224]  — la imatge transformada
        caption_tensor: LongTensor  [T]             — la caption com a llista d'índexs

    Nota important: cada imatge té 5 captions al CSV, per tant apareix 5 vegades
    al dataset. Cada aparició és una mostra independent (imatge diferent? No,
    la mateixa imatge, però emparellada amb una caption diferent).
    Flickr8k: 8.091 imatges × 5 captions = 40.455 mostres en total.
    """

    def __init__(
        self,
        images_dir:   str | Path,        # carpeta on es troben les imatges .jpg
        captions_csv: str | Path,        # fitxer CSV amb les captions
        vocab:        Vocabulary,        # vocabulari per codificar les captions
        transform=None,                  # transformacions d'imatge (si None → mode test)
        image_ids:    list[str] | None = None,  # llista de noms d'imatge per filtrar
                                                # (si None → usa totes les imatges)
    ):
        self.images_dir = Path(images_dir)
        self.vocab      = vocab
        # Si no es passa cap transformació, usem la de validació (sense aleatorietat)
        self.transform  = transform if transform is not None else get_transform(train=False)

        df = load_captions_df(captions_csv)  # llegeix el CSV → DataFrame [image, caption]

        if image_ids is not None:
            # Filtrem el DataFrame per quedar-nos NOMÉS amb les imatges de la llista.
            # isin() retorna True/False per cada fila; el filtre selecciona les True.
            # set(image_ids) converteix la llista a conjunt per a cerca O(1).
            df = df[df["image"].isin(set(image_ids))].reset_index(drop=True)

        self.df = df  # guardem el DataFrame filtrat

    def __len__(self) -> int:
        """Retorna el nombre total de mostres (files del DataFrame)."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Retorna la mostra número idx com a parell (tensor_imatge, tensor_caption).

        El DataLoader crida aquest mètode automàticament per a cada mostra
        del batch, en paral·lel si num_workers > 0.

        Args:
            idx: posició de la mostra (de 0 a len(dataset)-1)

        Returns:
            image:   FloatTensor [3, 224, 224]  — imatge normalitzada
            ids:     LongTensor  [T]             — caption com a índexs del vocabulari
        """
        row     = self.df.iloc[idx]         # fila número idx del DataFrame
        img_name = row["image"]             # nom del fitxer: "3145838052_1a54fc54f6.jpg"
        caption  = str(row["caption"])      # text de la caption: "a dog runs on the grass"

        # Obre la imatge des del disc i assegura que té 3 canals de color (RGB).
        # .convert("RGB") és necessari perquè algunes imatges poden ser en escala de grisos (1 canal) o tenir canal alfa de transparència (4 canals RGBA).
        image = Image.open(self.images_dir / img_name).convert("RGB")

        # Aplica les transformacions: PIL Image → tensor [3, 224, 224] normalitzat
        image = self.transform(image)

        # Converteix la caption de text a llista d'índexs enters, amb <start> i <end>
        # Exemple: "a dog" → [1, 8, 4, 2]  on 1=<start>, 8="a", 4="dog", 2=<end>
        ids = self.vocab.encode(caption, add_special=True)

        # torch.long és el tipus esperat per les capes nn.Embedding (accepten enters, no flotants)
        return image, torch.tensor(ids, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# COLLATE FUNCTION (preparació dels batches)
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Agrupa una llista de mostres individuals en un batch uniforme.

    El problema: les captions del batch tenen longituds diferents.
    Ex:  mostra 1: caption de 12 tokens
         mostra 2: caption de  7 tokens
         mostra 3: caption de 15 tokens
    No podem apilar tensors de mides diferents directament.

    La solució: PADDING — afegim zeros al final de les captions curtes fins que
    totes tinguin la longitud de la més llarga.
    El token 0 és <pad>, que el model aprendrà a ignorar.

    Returns:
        images:   FloatTensor [B, 3, 224, 224]  — batch d'imatges apilades
        targets:  LongTensor  [B, T_max]         — captions amb padding (T_max = la més llarga)
        lengths:  list[int]                      — longitud REAL de cada caption (sense padding)

    Args:
        batch: llista de tuples (image_tensor, caption_tensor) retornades per __getitem__
    """
    # Ordenem el batch de caption més llarga a més curta.
    # Necessari per a pack_padded_sequence a la LSTM, que espera aquest ordre.
    batch.sort(key=lambda x: len(x[1]), reverse=True)

    # Desempaquem el batch en dues llistes separades:
    # images = (tensor1, tensor2, ...), caps = (tensor1, tensor2, ...)
    images, caps = zip(*batch)

    # torch.stack apila una llista de tensors al llarg d'una nova dimensió.
    # [(3,224,224), (3,224,224), ...] → [B, 3, 224, 224]
    images = torch.stack(images, dim=0)

    # Guardem la longitud REAL de cada caption (incloent <start> i <end>)
    lengths = [len(c) for c in caps]
    # Ex: caps = [tensor([1,4,7,2]), tensor([1,5,3])] → lengths = [4, 3]

    # Creem la matriu de zeros [B, T_max] que omplirà de zeros (<pad>)
    # totes les posicions que no corresponen a tokens reals
    targets = torch.zeros(len(caps), max(lengths), dtype=torch.long)
    # torch.long perquè són índexs enters

    # Omplim cada fila amb els tokens reals de la caption corresponent.
    # Les posicions que sobren ja eren 0 (padding).
    for i, c in enumerate(caps):
        targets[i, : lengths[i]] = c
    # Exemple visual (T_max=5):
    # caps[0] = [1, 4, 7, 9, 2]  → targets[0] = [1, 4, 7, 9, 2]
    # caps[1] = [1, 5, 3, 2]     → targets[1] = [1, 5, 3, 2, 0]  ← 0 = <pad>
    # caps[2] = [1, 8, 2]        → targets[2] = [1, 8, 2, 0, 0]

    return images, targets, lengths

def collate_fn_scst(batch):
    """
    Igual que collate_fn, però cada mostra també inclou un identificador d'imatge.

    En entrenament SCST (Self-Critical Sequence Training) necessitem saber
    quina caption pertany a quina imatge original per poder calcular mètriques 
    comparant les captions generades amb les referències reals de cada imatge.

    Cada element del batch té la forma:
        (image_tensor, caption_tensor, img_id)

    Returns:
        images:   FloatTensor [B, 3, 224, 224]  — batch d'imatges apilades
        targets:  LongTensor  [B, T_max]        — captions amb padding
        lengths:  list[int]                     — longitud REAL de cada caption
        img_ids:  list[str]                     — identificadors originals de les imatges

    Args:
        batch: llista de tuples (image_tensor, caption_tensor, img_id)
    """

    # Ordenem les captions de més llarga a més curta.
    # Igual que a collate_fn, és necessari per pack_padded_sequence.
    batch.sort(key=lambda x: len(x[1]), reverse=True)

    # Separem cada component del batch en llistes independents.
    # images  = (img1, img2, ...)
    # caps    = (cap1, cap2, ...)
    # img_ids = ("123.jpg", "456.jpg", ...)
    images, caps, img_ids = zip(*batch)

    # Apilem totes les imatges en un únic tensor.
    # [(3,224,224), ...] → [B, 3, 224, 224]
    images = torch.stack(images, dim=0)

    # Longitud REAL de cada caption abans del padding.
    lengths = [len(c) for c in caps]

    # Creem la matriu [B, T_max] inicialitzada a zeros (<pad>)
    targets = torch.zeros(len(caps), max(lengths), dtype=torch.long)

    # Copiem cada caption dins la seva fila corresponent.
    # La resta de posicions continuen sent 0 (= padding)
    for i, c in enumerate(caps):
        targets[i, :lengths[i]] = c

    # Retornem també img_ids perquè SCST necessita saber a quina imatge correspon cada caption generada.
    return images, targets, lengths, list(img_ids)


def get_scst_loader(images_dir: str | Path, captions_csv: str | Path, vocab: Vocabulary, train_ids: list[str],
    batch_size: int = 16, num_workers: int = 2, image_size: int = 224) -> DataLoader:
    """
    Crea el DataLoader utilitzat durant l'entrenament SCST.

    Aquest DataLoader funciona igual que el de training normal,
    però cada batch també retorna els img_ids originals.

    Flux complet:
        Flickr8kDataset
            ↓
        __getitem__()
            retorna (image, caption, img_id)
            ↓
        collate_fn_scst()
            aplica padding i crea el batch
            ↓
        DataLoader
            retorna (images, targets, lengths, img_ids)

    Returns:
        DataLoader que produeix batches amb:
            images   → Tensor [B, 3, H, W]
            captions → Tensor [B, T_max]
            lengths  → longituds reals
            img_ids  → ids de les imatges

    Args:
        images_dir:    carpeta amb les imatges
        captions_csv:  fitxer CSV amb captions
        vocab:         vocabulari token ↔ índex
        train_ids:     llista d'imatges que formen el train split
        batch_size:    nombre de mostres per batch
        num_workers:   processos paral·lels per carregar dades
        image_size:    mida final de les imatges transformades
    """

    # Creem el dataset Flickr8k configurat per SCST.
    # return_image_id=True fa que __getitem__ també retorni img_id.
    ds = Flickr8kDataset(images_dir,captions_csv,vocab,transform=get_transform(image_size, train=True),image_ids=train_ids,return_image_id=True)

    # Emboliquem el dataset en un DataLoader que utilitza collate_fn_scst per preparar els batches.
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,           # barreja les mostres cada epoch
        num_workers=num_workers,     # càrrega paral·lela de dades
        collate_fn=collate_fn_scst,  # aplica padding a captions variables
        pin_memory=True,          # acceleració còpia CPU → GPU
    )

# ─────────────────────────────────────────────────────────────────────────────
# DIVISIÓ TRAIN / VAL / TEST
# ─────────────────────────────────────────────────────────────────────────────

def split_image_ids(captions_csv: str | Path, val_size: int = 1000, test_size: int = 1000, seed: int = 42):
    """
    Divideix les imatges úniques del dataset en tres conjunts disjunts.

    Per què dividir per IMATGES i no per captions?
    Si dividíssim per captions, podria passar que la caption 1 d'una imatge
    estigués al train i la caption 3 de la MATEIXA imatge estigués al test.
    El model hauria "vist" la imatge durant l'entrenament i les mètriques
    del test no reflectirien la capacitat de generalitzar a imatges noves.

    Per què la llavor fixa (seed=42)?
    Perquè tots els experiments usin EXACTAMENT la mateixa divisió.
    Sense seed fixa, cada execució barrejaria les imatges diferent i no
    podríem comparar experiments de forma justa.

    Flickr8k: 8.091 imatges úniques
        train: 6.091  (usades per entrenar el model)
        val:   1.000  (per avaluar i ajustar hiperparàmetres durant l'entrenament)
        test:  1.000  (avaluació final, NO es mira fins al final)

    Returns:
        train, val, test: llistes de noms de fitxers d'imatge
    """
    import numpy as np

    df     = load_captions_df(captions_csv)
    unique = sorted(df["image"].unique().tolist())
    # sorted() per garantir ordre determinista abans de barrejar, encara que la llavor fixa ja ho faria.

    rng = np.random.default_rng(seed)  # generador d'aleatorietat amb llavor fixa
    rng.shuffle(unique)                # barreja les imatges al lloc (in-place)

    # Primer les de test, després les de val, la resta per a train
    test  = unique[:test_size]
    val   = unique[test_size : test_size + val_size]
    train = unique[test_size + val_size :]

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# CREACIÓ DELS DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(images_dir: str | Path, captions_csv: str | Path, vocab: Vocabulary, batch_size: int = 32, num_workers: int = 2, image_size: int = 224):
    """
    Crea i retorna els tres DataLoaders (train, val, test) llestos per usar.

    Un DataLoader és un iterador que, en cada pas del bucle d'entrenament,
    retorna un batch de (images, captions, lengths) ja preparat per al model.

    Args:
        images_dir:   carpeta amb les imatges .jpg
        captions_csv: fitxer CSV de captions
        vocab:        vocabulari per codificar captions
        batch_size:   nombre de mostres per batch (32 és habitual)
        num_workers:  processos paral·lels per carregar imatges del disc.
                      0 → tot al thread principal (més lent però sense errors)
                      2-4 → paral·lel (si hi ha GPU)
        image_size:   mida de la imatge en píxels (224 per a ResNet)

    Returns:
        train_loader: DataLoader amb barreja i data augmentation
        val_loader:   DataLoader sense barreja ni augmentation
        test_loader:  DataLoader sense barreja ni augmentation
        (train_ids, val_ids, test_ids): les llistes de noms d'imatge de cada split
    """
    # Pas 1: Obtenim les llistes de noms d'imatge per a cada split
    train_ids, val_ids, test_ids = split_image_ids(captions_csv)

    # Pas 2: Creem un Dataset per a cada split.
    # Cada Dataset sap quines imatges li corresponen i quines transformacions aplicar.
    train_ds = Flickr8kDataset(
        images_dir, captions_csv, vocab,
        transform=get_transform(image_size, train=True),   # amb data augmentation
        image_ids=train_ids,
    )
    val_ds = Flickr8kDataset(
        images_dir, captions_csv, vocab,
        transform=get_transform(image_size, train=False),  # sense data augmentation
        image_ids=val_ids,
    )
    test_ds = Flickr8kDataset(
        images_dir, captions_csv, vocab,
        transform=get_transform(image_size, train=False),
        image_ids=test_ids,
    )

    # Pas 3: Emboliquem cada Dataset en un DataLoader.
    # El DataLoader s'encarrega de:
    #   - Cridar __getitem__ per a cada mostra del batch
    #   - Paral·lelitzar la càrrega (num_workers)
    #   - Cridar collate_fn per preparar el batch (padding, ordenació)
    #   - Moure els tensors a la memòria de la GPU (pin_memory)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,              # barreja les mostres a cada època → millor entrenament
        num_workers=num_workers,
        collate_fn=collate_fn,     # funció personalitzada per gestionar el padding
        pin_memory=True,           # carrega tensors a memòria fixada per accelerar la transferència CPU→GPU
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,             # no barregem: volem resultats reproduïbles
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, (train_ids, val_ids, test_ids)


# ─────────────────────────────────────────────────────────────────────────────
# FLUX RESUMIT D'AQUEST FITXER:
#
#  split_image_ids()         → llistes de noms d'imatge per split
#       ↓
#  Flickr8kDataset()         → objecte que sap donar mostres (imatge, caption) per índex
#       ↓
#  DataLoader()              → iterador que genera batches preparats automàticament
#       ↓
#  for images, caps, lens    → el que rep el bucle d'entrenament
#     in train_loader:       → images [B,3,224,224], caps [B,T], lens list[int]
# ─────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════
# FLICKR30K — HuggingFace (nlphuji/flickr30k)
# ════════════════════════════════════════════════════════

class Flickr30kHFDataset(Dataset):
    """
    Dataset que llegeix Flickr30k des del format HuggingFace (nlphuji/flickr30k).

    El dataset HF té 31014 files, cadascuna amb una imatge PIL i una llista de
    5 captions. Aquesta classe expandeix cada fila en 5 mostres individuals
    (una per caption), igual que fa Flickr8kDataset amb el CSV.

    Args:
        hf_split:   el split HF filtrat per 'train'/'val'/'test' (ja filtrat)
        vocab:      vocabulari Vocabulary
        transform:  transformació de torchvision (get_transform())
        return_image_id: si True, retorna també el filename de la imatge
    """

    def __init__(
        self,
        hf_split,
        vocab: Vocabulary,
        transform=None,
        return_image_id: bool = False,
    ):
        self.vocab = vocab
        self.transform = transform if transform is not None else get_transform(train=False)
        self.return_image_id = return_image_id

        # Expandeix: cada imatge té 5 captions → creem un índex pla (img_idx, cap_idx)
        self.samples: list[tuple[int, int]] = []
        self.hf_data = hf_split
        for img_idx in range(len(hf_split)):
            n_caps = len(hf_split[img_idx]["caption"])
            for cap_idx in range(n_caps):
                self.samples.append((img_idx, cap_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_idx, cap_idx = self.samples[idx]
        row = self.hf_data[img_idx]

        image = row["image"].convert("RGB")      # PIL Image ja carregada pel HF dataset
        image = self.transform(image)            # [3, 224, 224]

        caption = row["caption"][cap_idx]
        ids = self.vocab.encode(caption, add_special=True)

        if self.return_image_id:
            return image, torch.tensor(ids, dtype=torch.long), row["filename"]
        return image, torch.tensor(ids, dtype=torch.long)


def get_loaders_hf(
    hf_dataset,
    vocab: Vocabulary,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
):
    """Crea DataLoaders per Flickr30k des del dataset HuggingFace.

    Args:
        hf_dataset: resultat de load_dataset('nlphuji/flickr30k', ...) → conté la clau 'test'
                    (el dataset HF posa tot en un únic split 'test'; el camp 'split' indica
                    'train'/'val'/'test' de Karpathy)
        vocab:      vocabulari Vocabulary ja construït

    Returns:
        train_loader, val_loader, test_loader
    """
    full = hf_dataset["test"]  # totes les 31014 imatges

    # Filtra per split de Karpathy (camp 'split' dins cada fila)
    train_hf = full.filter(lambda x: x["split"] == "train")
    val_hf   = full.filter(lambda x: x["split"] == "val")
    test_hf  = full.filter(lambda x: x["split"] == "test")

    train_ds = Flickr30kHFDataset(train_hf, vocab, get_transform(image_size, train=True))
    val_ds   = Flickr30kHFDataset(val_hf,   vocab, get_transform(image_size, train=False))
    test_ds  = Flickr30kHFDataset(test_hf,  vocab, get_transform(image_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_scst_loader_hf(
    hf_dataset,
    vocab: Vocabulary,
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 224,
) -> DataLoader:
    """DataLoader SCST per Flickr30k HF — retorna (images, captions, lengths, img_ids)."""
    full = hf_dataset["test"]
    train_hf = full.filter(lambda x: x["split"] == "train")
    ds = Flickr30kHFDataset(train_hf, vocab, get_transform(image_size, train=True),
                            return_image_id=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, collate_fn=collate_fn_scst, pin_memory=True)