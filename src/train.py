"""Training script for the Flickr8k Image Captioning baseline.

Usage:
    python -m src.train --epochs 5 --batch-size 32 --wandb
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

# Add project root to sys.path for portability
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

try:
    from src.dataset import get_loaders
    from src.model import DecoderRNN, EncoderCNN
    from src.vocabulary import Vocabulary, build_vocab
except ImportError:
    from dataset import get_loaders
    from model import DecoderRNN, EncoderCNN
    from vocabulary import Vocabulary, build_vocab

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="data/flickr8k/Images")
    p.add_argument("--captions-csv", default="data/flickr8k/captions.txt")
    p.add_argument("--vocab-path", default="data/flickr8k/vocab.pkl")
    p.add_argument("--checkpoints-dir", default="checkpoints")
    p.add_argument("--vocab-threshold", type=int, default=5)

    p.add_argument("--embed-size", type=int, default=256)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--backbone", default="resnet50")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-step", type=int, default=20)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="image-captioning")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def get_or_build_vocab(args) -> Vocabulary:
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
    encoder.eval()
    decoder.eval()
    losses = []
    for images, captions, lengths in loader:
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True).data
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    vocab = get_or_build_vocab(args)
    print(f"[vocab] size = {len(vocab)}")

    train_loader, val_loader, _, _ = get_loaders(
        images_dir=args.images_dir,
        captions_csv=args.captions_csv,
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[data] train batches={len(train_loader)}  val batches={len(val_loader)}")

    encoder = EncoderCNN(args.embed_size, backbone=args.backbone).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    params = (
        list(decoder.parameters())
        + list(encoder.linear.parameters())
        + list(encoder.bn.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=args.lr)

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.config.update({"vocab_size": len(vocab)})

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        t0 = time.time()
        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            targets = pack_padded_sequence(captions, lengths, batch_first=True).data

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if i % args.log_step == 0:
                ppl = float(np.exp(min(loss.item(), 20)))
                print(f"epoch {epoch}/{args.epochs}  step {i}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  ppl={ppl:.2f}")
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/perplexity": ppl,
                               "epoch": epoch, "step": global_step})

        val_loss = evaluate(encoder, decoder, val_loader, criterion, device)
        val_ppl = float(np.exp(min(val_loss, 20)))
        elapsed = time.time() - t0
        print(f"== epoch {epoch} done  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  ({elapsed:.0f}s)")
        if use_wandb:
            wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "epoch": epoch})

        ckpt = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "vocab_size": len(vocab),
            "args": vars(args),
        }
        out = Path(args.checkpoints_dir) / f"ckpt_epoch{epoch}.pt"
        torch.save(ckpt, out)
        print(f"[ckpt] saved {out}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
