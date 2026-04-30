"""Training script for the Attention-based Image Captioning model.

Usage:
    python -m src.attention.train --epochs 10 --batch-size 32
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.shared.dataset import get_loaders
from src.attention.model import AttentionDecoder, EncoderCNNAttention
from src.shared.vocabulary import Vocabulary, build_vocab


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="data/flickr8k/Images")
    p.add_argument("--captions-csv", default="data/flickr8k/captions.txt")
    p.add_argument("--vocab-path", default="data/flickr8k/vocab.pkl")
    p.add_argument("--checkpoints-dir", default="checkpoints_attention")
    p.add_argument("--vocab-threshold", type=int, default=5)

    p.add_argument("--embed-size", type=int, default=256)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--attention-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--backbone", default="resnet50")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-step", type=int, default=20)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="image-captioning")
    p.add_argument("--wandb-entity", default=None)
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
        targets = pack_padded_sequence(
            captions[:, 1:], [l - 1 for l in lengths], batch_first=True
        ).data
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

    encoder = EncoderCNNAttention(backbone=args.backbone).to(device)
    decoder = AttentionDecoder(
        encoder_dim=encoder.encoder_dim,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(decoder.parameters()), lr=args.lr
    )

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.run_name, config=vars(args))
        wandb.config.update({"vocab_size": len(vocab)})

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        t0 = time.time()

        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            targets = pack_padded_sequence(
                captions[:, 1:], [l - 1 for l in lengths], batch_first=True
            ).data

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            train_losses.append(loss.item())
            if i % args.log_step == 0:
                ppl = float(np.exp(min(loss.item(), 20)))
                print(f"epoch {epoch}/{args.epochs}  step {i}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  ppl={ppl:.2f}")
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/perplexity": ppl,
                               "epoch": epoch, "step": global_step})

        val_loss = evaluate(encoder, decoder, val_loader, criterion, device)
        val_losses.append(val_loss)
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(ckpt, Path(args.checkpoints_dir) / "ckpt_best.pt")
            print(f"[early_stop] new best val_loss={best_val_loss:.4f} -> saved ckpt_best.pt")
        else:
            patience_counter += 1
            print(f"[early_stop] no improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"[early_stop] patience exhausted, stopping at epoch {epoch}")
                break

    steps_per_epoch = len(train_loader)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, alpha=0.6, label="train (per batch)")
    for e in range(1, args.epochs + 1):
        axes[0].axvline(e * steps_per_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("batch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Train loss (attention)")
    axes[0].legend()
    axes[1].plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy loss")
    axes[1].set_title("Val loss per epoch (attention)")
    axes[1].legend()
    plt.tight_layout()
    plot_path = Path(args.checkpoints_dir) / "loss_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[plot] saved {plot_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
