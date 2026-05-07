"""Flickr8k / Flickr30k Dataset and DataLoader (PyTorch).

Expected folder structure::

    Flickr8k:
        Images/         # all .jpg files
        captions.txt    # CSV: image,caption

    Flickr30k (CSV):
        flickr30k_images/   # all .jpg files
        results.csv         # CSV: image_name| comment_number| comment

    Flickr30k (HuggingFace):
        load_dataset('nlphuji/flickr30k', cache_dir=...) — imatges ja incloses
"""
from __future__ import annotations # per anotacions modernes com: list[str]

from pathlib import Path # per rutes de fitxers de manera neta

import pandas as pd # per llegir el CSV de captions
import torch # per crear tensors
from PIL import Image # per obrir imatges .jpg
from torch.utils.data import DataLoader, Dataset # dues classes de PyTorch
                # Dataset representa una col·lecció de dades
                # DataLoader agafa un Dataset i crea batches automàticament
from torchvision import transforms # importa transformacions d'imatge predefinides

from src.shared.vocabulary import Vocabulary # la classe Vocabulary del fitxer anterior


def load_captions_df(captions_csv: str | Path) -> pd.DataFrame:
    """Llegeix un CSV de captions i retorna sempre un DataFrame amb columnes 'image' i 'caption'.

    Suporta dos formats:
      - Flickr8k:  separador ',',  columnes: image, caption
      - Flickr30k: separador '|', columnes: image_name, comment_number, comment
    """
    captions_csv = Path(captions_csv)
    # Detecta format llegint la primera línia
    with open(captions_csv, encoding="utf-8") as f:
        header = f.readline()

    if "|" in header:  # Flickr30k
        df = pd.read_csv(captions_csv, sep="|", skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"image_name": "image", "comment": "caption"})
        df = df[["image", "caption"]].copy()
        df["image"] = df["image"].str.strip()
        df["caption"] = df["caption"].astype(str).str.strip()
    else:  # Flickr8k
        df = pd.read_csv(captions_csv)
        df = df[["image", "caption"]].copy()

    return df.reset_index(drop=True)


# Constants de normalització d'ImageNet (perquè l'encoder és un ResNet preentrenat amb ImageNet)
# les imatges s'han de ppreparar igual que les imatges amb què va ser entrenada la ResNet
IMAGENET_MEAN = (0.485, 0.456, 0.406) # la mitjana per canal (R, G, B)
IMAGENET_STD = (0.229, 0.224, 0.225) # la desviació típica per canal (R, G, B)
# la normalització fa això per cada píxel: (píxel - mean) / std
# (per això a stats.txt les imgs normalitzades poden tenir valors negatius i positius)

def get_transform(image_size: int = 224, train: bool = True): # retorna les transformacions que s'aplicaran
    if train: # durant l'entrenament, volem augmentar les dades amb transformacions aleatòries
        return transforms.Compose([ # aplica aquestes transformacions una darrere l'altra
            transforms.Resize(256), # redimensiona mantenint la proporció perquè el costat curt tingui 256 píxels
            transforms.RandomCrop(image_size), # retalla aleatòriament un quadrat de 224x224 píxels --> data augmentation (cada vegada pot veure un tros diferent de la imatge)
            transforms.RandomHorizontalFlip(), # gira la imatge horitzontalment 50% de les vegades --> gos mirant dreta o esquerra
            transforms.ToTensor(), # passem de PIL Image (amb valors de 0 - 255) a Tensor de PyTorch [3, 224, 224] amb valors entre 0 i 1
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), # normalitza cada canal amb la mitjana i desviació d'ImageNet
        ])                                                     # ara tenim valors centrats al voltant de 0 (ja no de 0 a 1)
    return transforms.Compose([ # en el cas de validació/test no fa transformaicions aleatòries, només les necessaries per preparar per la ResNet
        transforms.Resize(256),
        transforms.CenterCrop(image_size), # retalla el centre, no un tros aleatori (perquè volem resultats estables per a validacions diferents)
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Flickr8kDataset(Dataset):
    """Returns (image_tensor, caption_ids_tensor) per sample.

    Each row of the CSV is one (image, caption) pair, so an image with 5
    captions appears 5 times.
    """
    # es retorna un tensor d'imatge i un tensor d'ids de paraules per cada mostra
    # si la imatge té 5 captions, apareix 5 vegades al CSV, cada mostra és un (imatge, caption) diferent

    def __init__(
        self,
        images_dir: str | Path,
        captions_csv: str | Path,
        vocab: Vocabulary,
        transform=None,
        image_ids: list[str] | None = None,
        return_image_id: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.transform = transform if transform is not None else get_transform(train=False)
        # si no s'especifica cap transformació, per defecte no fa random crop ni flip, només les necessaries per la ResNet
        self.return_image_id = return_image_id

        df = load_captions_df(captions_csv)  # llegeix i normalitza Flickr8k o Flickr30k
        if image_ids is not None:
            df = df[df["image"].isin(set(image_ids))].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int): # rep un index i retorna la mostra en format (imatge_tensor, caption_numèrica_tensor)
        row = self.df.iloc[idx] # agafa la fila numero index
        img_name = row["image"] # nom de la imatge
        caption = str(row["caption"]) # caption d'aquella mostra (pot haver'n-hi 5)

        image = Image.open(self.images_dir / img_name).convert("RGB") # obre la imatge amb PIL i la força a tenir 3 canals
        image = self.transform(image) # aplica les transformacions. Ara tenim un tensor amb shape [3, 224, 224]

        ids = self.vocab.encode(caption, add_special=True) # converteix la caption a llista d'enters segons el vocabulari
        if self.return_image_id:
            return image, torch.tensor(ids, dtype=torch.long), img_name
        return image, torch.tensor(ids, dtype=torch.long) # retorna la imatge (en format tensor) i la caption en format tensor([1, 4, 27, 83, 2])
                                            # torch.long perquè les capes d'embedding esperen tensors de tipus long


def collate_fn(batch):
    """Pad captions to the longest in the batch and sort by length (descending).

    Returns:
        images:   FloatTensor [B, 3, H, W]
        captions: LongTensor  [B, T] padded with <pad>=0 ######### T és la longitud de la caption més llarga del batch
        lengths:  list[int]   original lengths (including <start>/<end>)
    """
    batch.sort(key=lambda x: len(x[1]), reverse=True) # Ordena el batch de captions més llargues a més curtes (RNN fucniona millor així??)
    images, caps = zip(*batch) # separa el batch en dues llistes: [(), (), ...] de tensors d'imatges i [(), (), ...] de tensors de captions
    images = torch.stack(images, dim=0) # apila les imatges en un unic tensor de shape [B, 3, 244, 244]

    lengths = [len(c) for c in caps] # llista amb la longitud de cada caption (amb START i END) caps=[tensor([1,4,7,2]),tensor([1,5,3])] --> lengths=[4,3]
    targets = torch.zeros(len(caps), max(lengths), dtype=torch.long) # crea un tensor de zeros de shape [B, T] on B és el num de captions i T la len de la caption més llarga (0 representa el token <pad>)
    for i, c in enumerate(caps):
        targets[i, : lengths[i]] = c # omple cada fila de targets amb els ids de la caption.
    return images, targets, lengths


def collate_fn_scst(batch):
    """Like collate_fn but each sample is (image, caption, img_id). Returns img_ids too."""
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, caps, img_ids = zip(*batch)
    images = torch.stack(images, dim=0)
    lengths = [len(c) for c in caps]
    targets = torch.zeros(len(caps), max(lengths), dtype=torch.long)
    for i, c in enumerate(caps):
        targets[i, :lengths[i]] = c
    return images, targets, lengths, list(img_ids)


def get_scst_loader(
    images_dir: str | Path,
    captions_csv: str | Path,
    vocab: Vocabulary,
    train_ids: list[str],
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 224,
) -> DataLoader:
    """DataLoader for SCST training — returns (images, captions, lengths, img_ids)."""
    ds = Flickr8kDataset(
        images_dir, captions_csv, vocab,
        transform=get_transform(image_size, train=True),
        image_ids=train_ids,
        return_image_id=True,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn_scst, pin_memory=True,
    )


def split_image_ids(captions_csv: str | Path, val_size: int = 1000, test_size: int = 1000, seed: int = 42):
    """Split unique image filenames into train/val/test (Karpathy-style)."""
    import numpy as np

    df = load_captions_df(captions_csv)
    unique = sorted(df["image"].unique().tolist()) # divideix per imatges úniques, no per captions (si no una caption estaria al train i laltra al test)
    rng = np.random.default_rng(seed) # generador aleatòri de numpy amb llavor fixa (sempre obtindrem la mateixa divisió --> bo per comparar experiments)
    rng.shuffle(unique) # barrega imatges

    test = unique[:test_size] # fa les particions
    val = unique[test_size : test_size + val_size] 
    train = unique[test_size + val_size :]
    return train, val, test # per defecte hi ha 8091 imatges: 6091 al train, 1.000 al val i 1.000 al test


def get_loaders(
    images_dir: str | Path,
    captions_csv: str | Path,
    vocab: Vocabulary,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
):
    train_ids, val_ids, test_ids = split_image_ids(captions_csv)

    train_ds = Flickr8kDataset(images_dir, captions_csv, vocab,
                               transform=get_transform(image_size, train=True),
                               image_ids=train_ids)
    val_ds = Flickr8kDataset(images_dir, captions_csv, vocab,
                             transform=get_transform(image_size, train=False),
                             image_ids=val_ids)
    test_ds = Flickr8kDataset(images_dir, captions_csv, vocab,
                              transform=get_transform(image_size, train=False),
                              image_ids=test_ids)
    # DataLoaders són iteradors que donen batches. Donem els datasets, la funció collate_fn per preparar els batches (pad de captions i ordenar per longitud), i altres paràmetres.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, # al train barrejem les mostres però al val i al test no cal.
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True) # pin_memory=True fa que els tensors es carreguin directament a la GPU (si està disponible) en lloc de passar per la RAM, per accelerar l'entrenament
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader, test_loader, (train_ids, val_ids, test_ids)

# Flux:
# 1. split_image_ids() fa la divisió de les imatges en train/val/test
# 2. crear datasets Flickr8kDataset per cada split --> donen les mostres (imatge_tensor, caption_ids_tensor)
# 3. crear DataLoaders per cada split --> donen batches preparats (images_tensor, captions_tensor, lengths) on captions_tensor està paddejat i ordenat per longitud


# ════════════════════════════════════════════════════════
# FLICKR30K — HuggingFace (nlphuji/flickr30k)
# ════════════════════════════════════════════════════════

class Flickr30kHFDataset(Dataset):
    """Dataset que llegeix Flickr30k des del format HuggingFace (nlphuji/flickr30k).

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
