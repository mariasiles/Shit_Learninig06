"""Training script for the Attention-based Image Captioning model.

Usage:
    python -m src.attention.train --epochs 10 --batch-size 32
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import pandas as pd  # per llegir el CSV de captions durant l'avaluació BLEU
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction  # mètriques BLEU
from nltk.translate.meteor_score import meteor_score  # mètrica METEOR (té en compte sinònims)

from src.shared.dataset import get_loaders, split_image_ids, load_captions_df  # dataloaders i divisió del dataset
from src.attention.model import AttentionDecoder, EncoderCNNAttention
from src.attention.sample import caption_image  # per generar captions durant l'avaluació BLEU
from src.shared.vocabulary import Vocabulary, build_vocab, simple_tokenize, load_glove_weights, load_word2vec_weights  # vocabulari i tokenitzador
from src.shared.losses import SemanticCrossEntropyLoss, build_soft_labels  # loss semàntica
from src.shared.metrics import rouge_l_score # mètriques compartides


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="dataset/Images")
    p.add_argument("--captions-csv", default="dataset/captions.txt")
    p.add_argument("--vocab-path", default="dataset/vocab.pkl")
    p.add_argument("--checkpoints-dir", default="checkpoints_attention")
    p.add_argument("--vocab-threshold", type=int, default=5)

    p.add_argument("--embed-size", type=int, default=256)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--attention-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--backbone", default="resnet152")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-step", type=int, default=20)

    p.add_argument("--glove-path", default=None,
                   help="Ruta al fitxer GloVe. Si s'especifica, activa la loss semàntica i inicialitza embeddings.")
    p.add_argument("--word2vec-path", default=None,
                   help="Ruta al fitxer Word2Vec (.bin binari o .txt text amb capçalera). "
                        "S'ignora si --glove-path també s'especifica.")
    p.add_argument("--word2vec-binary", action="store_true",
                   help="Indica que el fitxer Word2Vec és en format binari (.bin). "
                        "Si no s'activa, es detecta automàticament per l'extensió.")
    p.add_argument("--freeze-embeddings", action="store_true",
                   help="Si s'activa, els pesos (GloVe o Word2Vec) no s'actualitzen durant l'entrenament.")
    p.add_argument("--semantic-temp", type=float, default=10.0,
                   help="Temperatura pels soft labels semàntics (amb --glove-path o --word2vec-path).")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="image-captioning")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--semantic-loss", action="store_true",
                   help="Activa la funció de pèrdua semàntica usant similitud d'embeddings.")
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


@torch.no_grad()
def evaluate_bleu_subset(encoder, decoder, vocab, device, captions_csv, images_dir, n_samples=50) -> tuple[float, float, float, float]:
    encoder.eval()
    decoder.eval()
    _, _, test_ids = split_image_ids(captions_csv)
    df_caps = load_captions_df(captions_csv)
    
    sample_ids = test_ids[:n_samples]
    
    all_refs, all_hyps = [], []
    all_meteors = []
    all_rouges = []
    for img in sample_ids:
        refs = [simple_tokenize(c) for c in df_caps[df_caps["image"] == img]["caption"].tolist()]
        hyp = simple_tokenize(caption_image(f"{images_dir}/{img}", encoder, decoder, vocab, device))
        all_refs.append(refs)
        all_hyps.append(hyp)
        all_meteors.append(meteor_score(refs, hyp))
        all_rouges.append(rouge_l_score(refs, hyp))
        
    b1 = corpus_bleu(all_refs, all_hyps, weights=(1,0,0,0))
    b4 = corpus_bleu(all_refs, all_hyps, weights=(.25,.25,.25,.25))
    m = float(np.mean(all_meteors))
    r = float(np.mean(all_rouges))
    return float(b1), float(b4), m, r


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

    pretrained_weights = None
    if args.glove_path:
        pretrained_weights, glove_dim = load_glove_weights(args.glove_path, vocab)
        pretrained_weights = pretrained_weights.to(device)
        args.embed_size = glove_dim
        emb_type = "glove-frozen" if args.freeze_embeddings else "glove-finetune"
    elif args.word2vec_path:
        binary = args.word2vec_binary if args.word2vec_binary else None  # None → auto-detect per extensió
        pretrained_weights, w2v_dim = load_word2vec_weights(args.word2vec_path, vocab, binary=binary)
        pretrained_weights = pretrained_weights.to(device)
        args.embed_size = w2v_dim
        emb_type = "word2vec-frozen" if args.freeze_embeddings else "word2vec-finetune"
    else:
        emb_type = "scratch"
    print(f"[embeddings] tipus={emb_type}  embed_size={args.embed_size}")

    decoder = AttentionDecoder(
        encoder_dim=encoder.encoder_dim,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        pretrained_weights=pretrained_weights,
        freeze_embeddings=args.freeze_embeddings,
    ).to(device)

    if pretrained_weights is not None and args.semantic_loss:
        soft_lbls = build_soft_labels(decoder.embed.weight.data.cpu(), temperature=args.semantic_temp)
        criterion = SemanticCrossEntropyLoss(soft_lbls).to(device)
        print(f"[loss] SemanticCrossEntropy (temp={args.semantic_temp}) — soft labels des de {emb_type}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[loss] CrossEntropyLoss estàndard")
    optimizer = torch.optim.Adam(
        list(decoder.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
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
        scheduler.step(val_loss)
        val_ppl = float(np.exp(min(val_loss, 20)))
        
        # Avaluació ràpida de mètriques sobre un subconjunt (per no alentir la epoch)
        val_b1, val_b4, val_m, val_r = evaluate_bleu_subset(encoder, decoder, vocab, device, args.captions_csv, args.images_dir, n_samples=50)

        elapsed = time.time() - t0
        print(f"== epoch {epoch} done  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  val_bleu4={val_b4:.3f}  val_rouge={val_r:.3f} ({elapsed:.0f}s)")
        if use_wandb:
            wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "val/bleu1": val_b1, "val/bleu4": val_b4, "val/meteor": val_m, "val/rouge": val_r, "epoch": epoch,
                       "lr": optimizer.param_groups[0]["lr"]})

        ckpt = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "vocab_size": len(vocab),
            "args": vars(args),
        }
        out = Path(args.checkpoints_dir) / "ckpt_last.pt" # sobreescriu l'últim per estalviar espai
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

    # --- BLEU + METEOR evaluation on test set (millor checkpoint) ---
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    print("\n[bleu+meteor] avaluant sobre el conjunt de test...")
    best_ckpt = torch.load(Path(args.checkpoints_dir) / "ckpt_best.pt", map_location=device)
    encoder.load_state_dict(best_ckpt["encoder"])  # carrega pesos del millor model
    decoder.load_state_dict(best_ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    _, _, test_ids = split_image_ids(args.captions_csv)  # agafa els IDs del test set
    df_caps = load_captions_df(args.captions_csv)  # llegeix totes les captions (Flickr8k o Flickr30k)
    smooth = SmoothingFunction().method1  # suavitzat per evitar BLEU-4 = 0

    all_refs, all_hyps = [], []
    all_meteors = []
    bleu_table = wandb.Table(columns=["image", "generated_caption", "reference_captions", "BLEU-1", "BLEU-4", "METEOR"]) if use_wandb else None
    images_dir_abs = Path(args.images_dir).resolve()
    TABLE_LIMIT = 200

    print(f"Evaluating {len(test_ids)} test images...")
    print(f"{'Image':<35} {'BLEU-1':>7} {'BLEU-4':>7} {'METEOR':>7}  Caption")
    print("-" * 110)
    for img in test_ids:
        refs = [simple_tokenize(c) for c in df_caps[df_caps["image"] == img]["caption"].tolist()]
        hyp  = simple_tokenize(caption_image(f"{args.images_dir}/{img}", encoder, decoder, vocab, device))
        b1 = sentence_bleu(refs, hyp, weights=(1,0,0,0), smoothing_function=smooth)
        b4 = sentence_bleu(refs, hyp, weights=(.25,.25,.25,.25), smoothing_function=smooth)
        m  = meteor_score(refs, hyp)
        all_refs.append(refs)
        all_hyps.append(hyp)
        all_meteors.append(m)
        print(f"{img:<35} {b1:>7.3f} {b4:>7.3f} {m:>7.3f}  {' '.join(hyp)}")
        if bleu_table is not None and len(bleu_table.data) < TABLE_LIMIT:
            ref_str = " | ".join([" ".join(r) for r in refs])
            bleu_table.add_data(wandb.Image(str(images_dir_abs / img)), " ".join(hyp), ref_str, round(b1, 3), round(b4, 3), round(m, 3))

    cb1 = corpus_bleu(all_refs, all_hyps, weights=(1,0,0,0))
    cb4 = corpus_bleu(all_refs, all_hyps, weights=(.25,.25,.25,.25))
    cm  = float(np.mean(all_meteors))
    print("-" * 110)
    print(f"[bleu] Corpus BLEU-1: {cb1:.3f}  BLEU-4: {cb4:.3f}  METEOR: {cm:.3f}")

    if use_wandb:
        wandb.log({"bleu/corpus_bleu1": cb1, "bleu/corpus_bleu4": cb4, "bleu/meteor": cm, "bleu/eval_table": bleu_table})
    # ------------------------------------------------

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
