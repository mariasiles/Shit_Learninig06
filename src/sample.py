"""Generate a caption for a single image using a trained checkpoint."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Add project root to sys.path for portability
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from PIL import Image
from torchvision import transforms

try:
    from src.dataset import get_transform
    from src.model import DecoderRNN, EncoderCNN
    from src.vocabulary import Vocabulary
except ImportError:
    from dataset import get_transform
    from model import DecoderRNN, EncoderCNN
    from vocabulary import Vocabulary  # noqa: F401  (needed for pickle load)


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
    p.add_argument("--image", required=True, help="Ruta a la imatge a processar")
    p.add_argument("--checkpoint", default=None, help="Ruta al checkpoint .pt (per defecte l'últim)")
    p.add_argument("--vocab", default=str(root_dir / "data/flickr8k/vocab.pkl"), help="Ruta al vocabulari .pkl")
    args = p.parse_args()

    # Si no s'ha especificat checkpoint, busquem l'últim
    if args.checkpoint is None:
        ckpt_dir = root_dir / "checkpoints"
        ckpts = sorted(list(ckpt_dir.glob("*.pt")), key=lambda x: x.stat().st_mtime)
        if ckpts:
            args.checkpoint = str(ckpts[-1])
            print(f"[auto] Usant l'últim checkpoint trobat: {args.checkpoint}")
        else:
            print("ERROR: No s'ha trobat cap checkpoint a 'checkpoints/'")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, vocab = load_checkpoint(args.checkpoint, args.vocab, device)
    cap = caption_image(args.image, encoder, decoder, vocab, device)
    print(f"\nResultat:")
    print(f"{Path(args.image).name}: {cap}")


if __name__ == "__main__":
    main()
