"""Generate a caption for a single image using a trained checkpoint."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image

from src.dataset import get_transform
from src.model import DecoderRNN, EncoderCNN
from src.vocabulary import Vocabulary  # noqa: F401  (needed for pickle load)


def load_checkpoint(ckpt_path: str, vocab_path: str, device: torch.device):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    encoder = EncoderCNN(a["embed_size"], backbone=a["backbone"]).to(device).eval()
    decoder = DecoderRNN(a["embed_size"], a["hidden_size"], len(vocab),
                         a["num_layers"]).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    return encoder, decoder, vocab


@torch.no_grad()
def caption_image(image_path: str, encoder, decoder, vocab, device) -> str:
    tfm = get_transform(train=False)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    feat = encoder(x)
    ids = decoder.sample(feat).cpu().numpy()[0].tolist()
    return vocab.decode(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", default="data/flickr8k/vocab.pkl")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, vocab = load_checkpoint(args.checkpoint, args.vocab, device)
    cap = caption_image(args.image, encoder, decoder, vocab, device)
    print(f"{Path(args.image).name}: {cap}")


if __name__ == "__main__":
    main()
