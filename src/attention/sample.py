"""Generate a caption for a single image using the attention model with beam search."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image

from src.shared.dataset import get_transform
from src.attention.model import AttentionDecoder, EncoderCNNAttention
from src.shared.vocabulary import Vocabulary  # noqa: F401  (needed for pickle load)


def load_checkpoint(ckpt_path: str, vocab_path: str, device: torch.device):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    encoder = EncoderCNNAttention(backbone=a["backbone"]).to(device).eval()
    decoder = AttentionDecoder(
        encoder_dim=2048,
        embed_size=a["embed_size"],
        hidden_size=a["hidden_size"],
        vocab_size=len(vocab),
        attention_dim=a["attention_dim"],
        dropout=a["dropout"],
    ).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    return encoder, decoder, vocab


@torch.no_grad()
def caption_image(image_path: str, encoder, decoder, vocab, device, beam_size: int = 3) -> str:
    tfm = get_transform(train=False)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    features = encoder(x)                    # [1, 49, 2048]
    ids = decoder.beam_search(
        features,
        start_idx=vocab.word2idx["<start>"],
        end_idx=vocab.word2idx["<end>"],
        beam_size=beam_size,
    )
    return vocab.decode(ids, skip_special=False)


@torch.no_grad()
def caption_pil_image(pil_img, encoder, decoder, vocab, device, beam_size: int = 3) -> str:
    """Com caption_image però rep un PIL.Image directament (per al dataset HuggingFace)."""
    tfm = get_transform(train=False)
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    features = encoder(x)
    ids = decoder.beam_search(
        features,
        start_idx=vocab.word2idx["<start>"],
        end_idx=vocab.word2idx["<end>"],
        beam_size=beam_size,
    )
    return vocab.decode(ids, skip_special=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", default="data/flickr8k/vocab.pkl")
    p.add_argument("--beam-size", type=int, default=3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, vocab = load_checkpoint(args.checkpoint, args.vocab, device)
    cap = caption_image(args.image, encoder, decoder, vocab, device, beam_size=args.beam_size)
    print(f"{Path(args.image).name}: {cap}")


if __name__ == "__main__":
    main()
