"""Flickr8k Dataset and DataLoader (PyTorch).

Expected folder structure::

    data/flickr8k/
        Images/         # all .jpg files
        captions.txt    # CSV with header: image,caption
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.vocabulary import Vocabulary


# Standard ImageNet normalization (because the encoder is a ResNet pretrained on ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transform(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Flickr8kDataset(Dataset):
    """Returns (image_tensor, caption_ids_tensor) per sample.

    Each row of the CSV is one (image, caption) pair, so an image with 5
    captions appears 5 times.
    """

    def __init__(
        self,
        images_dir: str | Path,
        captions_csv: str | Path,
        vocab: Vocabulary,
        transform=None,
        image_ids: list[str] | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.transform = transform if transform is not None else get_transform(train=False)

        df = pd.read_csv(captions_csv)
        if image_ids is not None:
            df = df[df["image"].isin(set(image_ids))].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row["image"]
        caption = str(row["caption"])

        image = Image.open(self.images_dir / img_name).convert("RGB")
        image = self.transform(image)

        ids = self.vocab.encode(caption, add_special=True)
        return image, torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    """Pad captions to the longest in the batch and sort by length (descending).

    Returns:
        images:   FloatTensor [B, 3, H, W]
        captions: LongTensor  [B, T] padded with <pad>=0
        lengths:  list[int]   original lengths (including <start>/<end>)
    """
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, caps = zip(*batch)
    images = torch.stack(images, dim=0)

    lengths = [len(c) for c in caps]
    targets = torch.zeros(len(caps), max(lengths), dtype=torch.long)
    for i, c in enumerate(caps):
        targets[i, : lengths[i]] = c
    return images, targets, lengths


def split_image_ids(captions_csv: str | Path, val_size: int = 1000, test_size: int = 1000, seed: int = 42):
    """Split unique image filenames into train/val/test (Karpathy-style)."""
    import numpy as np

    df = pd.read_csv(captions_csv)
    unique = sorted(df["image"].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    test = unique[:test_size]
    val = unique[test_size : test_size + val_size]
    train = unique[test_size + val_size :]
    return train, val, test


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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader, test_loader, (train_ids, val_ids, test_ids)
