"""Training script for the Attention-based Image Captioning model.

Usage:
    python -m src.attention.train --epochs 10 --batch-size 32
"""
from __future__ import annotations

import argparse
import contextlib
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd  # per llegir el CSV de captions durant l'avaluació BLEU
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction  # mètriques BLEU
from nltk.translate.meteor_score import meteor_score  # mètrica METEOR (té en compte sinònims)

from src.shared.dataset import get_loaders, get_scst_loader, split_image_ids, load_captions_df, get_loaders_hf, get_scst_loader_hf  # dataloaders i divisió del dataset
from src.attention.model import AttentionDecoder, EncoderCNNAttention
from src.attention.sample import caption_image, caption_pil_image  # per generar captions durant l'avaluació BLEU
from src.shared.vocabulary import Vocabulary, build_vocab, build_vocab_hf, simple_tokenize, load_glove_weights, load_word2vec_weights  # vocabulari i tokenitzador
from src.shared.losses import SemanticCrossEntropyLoss, build_soft_labels  # loss semàntica


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
    p.add_argument("--no-semantic-loss", action="store_true",
                   help="Usa CrossEntropyLoss estàndard fins i tot amb GloVe/Word2Vec (embeddings preentrenats però loss CE).")
    p.add_argument("--freeze-embeddings", action="store_true",
                   help="Si s'activa, els pesos (GloVe o Word2Vec) no s'actualitzen durant l'entrenament.")
    p.add_argument("--semantic-temp", type=float, default=10.0,
                   help="Temperatura pels soft labels semàntics (amb --glove-path o --word2vec-path).")
    p.add_argument("--finetune-cnn-epoch", type=int, default=None,
                   help="Epoch a partir de la qual es descongela layer4 de la CNN (None = mai).")
    p.add_argument("--ds-lambda", type=float, default=1.0,
                   help="Pes de la Doubly Stochastic Attention regularització. 0 = desactivada.")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing per CrossEntropyLoss (0.0 = desactivat, 0.1 recomanat).")
    p.add_argument("--resume-from", default=None,
                   help="Checkpoint des del qual continuar l'entrenament (ex: checkpoints/ckpt_epoch14.pt).")

    # ── Flickr30k HuggingFace ──────────────────────────────────────────────
    p.add_argument("--flickr30k-hf", action="store_true",
                   help="Usa el dataset Flickr30k de HuggingFace (nlphuji/flickr30k) en lloc del CSV.")
    p.add_argument("--flickr30k-hf-cache", default="dataset/flickr30k_hf",
                   help="Carpeta cache del dataset HuggingFace.")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="image-captioning")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--run-name", default=None)

    # ── SCST amb CIDEr-D ───────────────────────────────────────────────────
    p.add_argument("--scst-epochs", type=int, default=0,
                   help="Nombre d'èpoques SCST (CIDEr-D reward) després del CE. 0 = sense SCST.")
    p.add_argument("--scst-lr", type=float, default=5e-5,
                   help="LR per a la fase SCST. Ha de ser molt baix (1e-5 a 1e-4).")
    p.add_argument("--scst-batch-size", type=int, default=16,
                   help="Batch size per a SCST (recomanat <= CE batch size per memòria).")
    p.add_argument("--scst-max-len", type=int, default=20,
                   help="Longitud màxima de les seqüències mostrejades durant SCST.")
    p.add_argument("--scst-checkpoint", default=None,
                   help="Checkpoint inicial per SCST. Si s'especifica i --epochs 0, salta el CE.")
    p.add_argument("--scst-checkpoints-dir", default="checkpoints_attention_scst",
                   help="Carpeta on guardar els checkpoints SCST.")
    return p.parse_args()


def safe_save(obj, path, retries: int = 5):
    """torch.save amb reintents per errors NFS transitòris."""
    import shutil, time
    path = Path(path)
    for attempt in range(retries):
        try:
            tmp = path.parent / f".tmp_{path.name}"
            torch.save(obj, tmp)
            shutil.move(str(tmp), str(path))
            return
        except RuntimeError:
            if attempt < retries - 1:
                print(f"[ckpt] error NFS (intent {attempt+1}/{retries}), reintentant...")
                time.sleep(3)
            else:
                raise


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
        outputs, _ = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
    return float(np.mean(losses))


# ════════════════════════════════════════════════════════
# CIDEr-D helpers
# ════════════════════════════════════════════════════════

def _get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def compute_idf_weights(df_train: "pd.DataFrame", n_max: int = 4) -> dict:
    """Pre-compute IDF weights from training captions (grouped by image).

    Returns {n: {ngram_tuple: idf_float}} for n in 1..n_max.
    Called once before the SCST loop — O(|train|) pre-processing.
    """
    images = df_train["image"].unique()
    N = len(images)

    idf: dict[int, dict] = {}
    for n in range(1, n_max + 1):
        df_cnt: dict = defaultdict(int)  # document frequency per n-gram
        for img in images:
            caps = df_train[df_train["image"] == img]["caption"].tolist()
            seen: set = set()
            for cap in caps:
                for ng in _get_ngrams(simple_tokenize(str(cap)), n):
                    seen.add(ng)
            for ng in seen:
                df_cnt[ng] += 1
        # log((N+1)/(df+1)) — Laplace-smoothed IDF
        idf[n] = {ng: math.log((N + 1) / (cnt + 1)) for ng, cnt in df_cnt.items()}

    print(f"[CIDEr-D] IDF computed from {N} training images")
    return idf


def cider_d_score(
    hyp_tokens: list[str],
    ref_tokens_list: list[list[str]],
    idf: dict,
    n_max: int = 4,
) -> float:
    """CIDEr-D score for one hypothesis vs a list of references.

    Uses n-gram TF-IDF cosine similarity with CIDEr-D clipping (clip hypothesis
    n-gram count by max reference count to penalise repetition).
    Scaled by 10 and averaged over orders 1-4, matching the original paper.
    """
    if not hyp_tokens or not ref_tokens_list:
        return 0.0

    total = 0.0
    for n in range(1, n_max + 1):
        idf_n = idf.get(n, {})
        h_ngrams = _get_ngrams(hyp_tokens, n)
        ref_ngrams_list = [_get_ngrams(ref, n) for ref in ref_tokens_list]

        if not h_ngrams:
            continue

        # CIDEr-D clipping
        h_clipped = {ng: min(cnt, max((r.get(ng, 0) for r in ref_ngrams_list), default=0))
                     for ng, cnt in h_ngrams.items()}

        len_h = max(len(hyp_tokens) - n + 1, 1)

        h_norm_sq = sum(
            ((h_clipped.get(ng, 0) / len_h) * idf_n.get(ng, 0.0)) ** 2
            for ng in h_clipped
        )
        h_norm = math.sqrt(h_norm_sq + 1e-10)

        ref_sum = 0.0
        for ref_tokens, ref_ngrams in zip(ref_tokens_list, ref_ngrams_list):
            len_r = max(len(ref_tokens) - n + 1, 1)
            dot = sum(
                (h_clipped[ng] / len_h) * idf_n.get(ng, 0.0)
                * (ref_ngrams[ng] / len_r) * idf_n.get(ng, 0.0)
                for ng in h_clipped
                if ng in ref_ngrams
            )
            r_norm = math.sqrt(sum(
                ((cnt / len_r) * idf_n.get(ng, 0.0)) ** 2
                for ng, cnt in ref_ngrams.items()
            ) + 1e-10)
            ref_sum += dot / (h_norm * r_norm)

        total += ref_sum / len(ref_tokens_list)

    return total * 10.0 / n_max


def scst_epoch(
    encoder: "EncoderCNNAttention",
    decoder: "AttentionDecoder",
    loader,
    optimizer: "torch.optim.Optimizer",
    vocab: "Vocabulary",
    refs_by_image: dict,
    idf: dict,
    device: "torch.device",
    args,
    use_wandb: bool = False,
    epoch: int = 0,
) -> float:
    """One SCST epoch using CIDEr-D as reward.

    For each image: sample a sequence + greedy decode; reward = CIDEr(sample)
    - CIDEr(greedy); REINFORCE loss = -reward * sum(log_probs of sampled tokens).
    Rewards are normalised per batch (zero-mean, unit-std) for training stability.
    """
    encoder.eval()   # encoder stays frozen during SCST
    decoder.train()

    start_idx = vocab.word2idx["<start>"]
    end_idx   = vocab.word2idx["<end>"]

    total_reward = 0.0
    n_batches = 0

    for batch_idx, (images, captions, lengths, img_ids) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        with torch.no_grad():
            features = encoder(images)   # [B, 49, 2048]  — encoder always frozen

        # Batched sample + greedy — processes all images in the batch in parallel
        sampled_tokens_list, log_probs_list = decoder.sample_batch_with_logprobs(
            features, start_idx, end_idx, max_len=args.scst_max_len
        )
        greedy_tokens_list = decoder.greedy_batch(
            features, start_idx, end_idx, max_len=args.scst_max_len
        )

        raw_rewards: list[float] = []
        log_probs_all: list[torch.Tensor] = []

        for i, img_id in enumerate(img_ids):
            refs = refs_by_image.get(img_id, [])
            if not refs:
                continue
            refs_tok = [simple_tokenize(str(r)) for r in refs]

            sampled_words = [vocab.idx2word.get(t, "<unk>") for t in sampled_tokens_list[i]]
            greedy_words  = [vocab.idx2word.get(t, "<unk>") for t in greedy_tokens_list[i]]

            r_sample = cider_d_score(sampled_words, refs_tok, idf)
            r_greedy  = cider_d_score(greedy_words,  refs_tok, idf)

            raw_rewards.append(r_sample - r_greedy)
            log_probs_all.append(log_probs_list[i])

        if not raw_rewards:
            continue

        # Per-batch reward normalisation (reduces variance, improves stability)
        rewards_t = torch.tensor(raw_rewards, dtype=torch.float32, device=device)
        if rewards_t.std() > 1e-6:
            rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        # REINFORCE loss: accumulate then backprop once
        loss = sum(-r * lp.sum() for r, lp in zip(rewards_t, log_probs_all))
        loss = loss / len(raw_rewards)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()

        mean_r = float(torch.tensor(raw_rewards).mean())
        total_reward += mean_r
        n_batches += 1

        if batch_idx % 20 == 0:
            print(f"  [SCST] epoch {epoch}  batch {batch_idx}/{len(loader)}"
                  f"  mean_reward={mean_r:.4f}")
            if use_wandb:
                import wandb
                wandb.log({"scst/mean_reward": mean_r, "scst/epoch": epoch})

    return total_reward / max(n_batches, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # ── Carrega dataset i vocab (CSV o HuggingFace) ────────────────────────
    if args.flickr30k_hf:
        from datasets import load_dataset
        print("[data] carregant Flickr30k HuggingFace...")
        hf_ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True,
                             cache_dir=args.flickr30k_hf_cache)
        vp = Path(args.vocab_path)
        if vp.exists():
            with open(vp, "rb") as f:
                import pickle; vocab = pickle.load(f)
            print(f"[vocab] carregat de {vp} (size={len(vocab)})")
        else:
            print("[vocab] construint des de HF dataset...")
            vocab = build_vocab_hf(hf_ds, threshold=args.vocab_threshold)
            vp.parent.mkdir(parents=True, exist_ok=True)
            with open(vp, "wb") as f:
                import pickle; pickle.dump(vocab, f)
            print(f"[vocab] built and saved to {vp} (size={len(vocab)})")

        train_loader, val_loader, _ = get_loaders_hf(
            hf_ds, vocab, batch_size=args.batch_size, num_workers=args.num_workers)

        # Construeix train_ids / test_ids / df_caps per a SCST i BLEU
        full = hf_ds["test"]
        train_rows = full.filter(lambda x: x["split"] == "train")
        val_rows   = full.filter(lambda x: x["split"] == "val")
        test_rows  = full.filter(lambda x: x["split"] == "test")
        train_ids  = [r["filename"] for r in train_rows]
        val_ids    = [r["filename"] for r in val_rows]
        test_ids   = [r["filename"] for r in test_rows]
        # df_caps: una fila per (imatge, caption) — igual que el CSV
        records = []
        for r in full:
            for cap in r["caption"]:
                records.append({"image": r["filename"], "caption": cap})
        import pandas as _pd
        df_caps_hf = _pd.DataFrame(records)
        # dict filename → PIL image per al test (per BLEU sense path a disc)
        test_pil = {r["filename"]: r["image"] for r in test_rows}
    else:
        vocab = get_or_build_vocab(args)
        train_loader, val_loader, _, (train_ids, val_ids, test_ids) = get_loaders(
            images_dir=args.images_dir,
            captions_csv=args.captions_csv,
            vocab=vocab,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    print(f"[vocab] size = {len(vocab)}")
    print(f"[data] train batches={len(train_loader)}  val batches={len(val_loader)}")

    encoder = EncoderCNNAttention(backbone=args.backbone).to(device)

    # Si fem SCST-only (--epochs 0 + --scst-checkpoint), saltem la càrrega de GloVe/Word2Vec
    # perquè els embeddings ja estan dins el checkpoint — estalviem ~3 min de càrrega
    scst_only = (args.epochs == 0 and args.scst_checkpoint is not None)

    pretrained_weights = None
    if scst_only:
        # Llegim embed_size i hidden_size del checkpoint per construir l'arquitectura correcta
        _meta = torch.load(args.scst_checkpoint, map_location="cpu")["args"]
        args.embed_size  = _meta.get("embed_size",  args.embed_size)
        args.hidden_size = _meta.get("hidden_size", args.hidden_size)
        args.attention_dim = _meta.get("attention_dim", args.attention_dim)
        emb_type = "from_checkpoint"
        print(f"[SCST-only] saltant càrrega de GloVe/Word2Vec — embed_size={args.embed_size}")
    elif args.glove_path:
        pretrained_weights, glove_dim = load_glove_weights(args.glove_path, vocab)
        pretrained_weights = pretrained_weights.to(device)
        args.embed_size = glove_dim
        emb_type = "glove-frozen" if args.freeze_embeddings else "glove-finetune"
    elif args.word2vec_path:
        binary = args.word2vec_binary if args.word2vec_binary else None
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

    if pretrained_weights is not None and not args.no_semantic_loss:
        soft_lbls = build_soft_labels(decoder.embed.weight.data.cpu(), temperature=args.semantic_temp)
        criterion = SemanticCrossEntropyLoss(soft_lbls).to(device)
        print(f"[loss] SemanticCrossEntropy (temp={args.semantic_temp}) — soft labels des de {emb_type}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        ls_tag = f" (label_smoothing={args.label_smoothing})" if args.label_smoothing > 0 else ""
        print(f"[loss] CrossEntropyLoss estàndard{ls_tag}")
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
    start_epoch = 1

    if args.resume_from:
        print(f"[resume] carregant checkpoint: {args.resume_from}")
        res = torch.load(args.resume_from, map_location=device, weights_only=False)
        encoder.load_state_dict(res["encoder"])
        decoder.load_state_dict(res["decoder"])
        start_epoch = res["epoch"] + 1
        print(f"[resume] continuant des de l'epoch {res['epoch']} → inici epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        if args.finetune_cnn_epoch and epoch == args.finetune_cnn_epoch:
            encoder.finetuning = True
            for p in encoder.cnn[-1].parameters():
                p.requires_grad = True
            optimizer.add_param_group({"params": list(encoder.cnn[-1].parameters()), "lr": args.lr / 10})
            print(f"[finetune] epoch {epoch}: layer4 descongelada (lr={args.lr/10:.2e})")

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
            outputs, alphas_sum = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            if args.ds_lambda > 0:
                loss = loss + args.ds_lambda * ((1 - alphas_sum) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
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
        elapsed = time.time() - t0
        print(f"== epoch {epoch} done  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  ({elapsed:.0f}s)")
        if use_wandb:
            wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "epoch": epoch,
                       "lr": optimizer.param_groups[0]["lr"]})

        ckpt = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "vocab_size": len(vocab),
            "args": vars(args),
        }
        out = Path(args.checkpoints_dir) / f"ckpt_epoch{epoch}.pt"
        safe_save(ckpt, out)
        print(f"[ckpt] saved {out}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            safe_save(ckpt, Path(args.checkpoints_dir) / "ckpt_best.pt")
            print(f"[early_stop] new best val_loss={best_val_loss:.4f} -> saved ckpt_best.pt")
        else:
            patience_counter += 1
            print(f"[early_stop] no improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"[early_stop] patience exhausted, stopping at epoch {epoch}")
                break

    if train_losses:
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

    # ── SCST fine-tuning (CIDEr-D reward) ─────────────────────────────────
    # Point that BLEU evaluation will load from (updated after SCST if run)
    best_ckpt_path = Path(args.checkpoints_dir) / "ckpt_best.pt"

    if args.scst_epochs > 0:
        print("\n[SCST] Iniciant fine-tuning SCST amb CIDEr-D reward...")
        Path(args.scst_checkpoints_dir).mkdir(parents=True, exist_ok=True)

        # Determine which checkpoint to start from
        scst_start = args.scst_checkpoint or str(best_ckpt_path)
        if not Path(scst_start).exists():
            raise FileNotFoundError(
                f"[SCST] Checkpoint not found: {scst_start}\n"
                "Run CE training first, or pass --scst-checkpoint <path>."
            )
        print(f"[SCST] Carregant checkpoint: {scst_start}")
        scst_ckpt = torch.load(scst_start, map_location=device)
        encoder.load_state_dict(scst_ckpt["encoder"])
        decoder.load_state_dict(scst_ckpt["decoder"])
        # Ensure encoder is fully frozen for SCST stability
        encoder.finetuning = False
        for p in encoder.parameters():
            p.requires_grad = False

        # Pre-compute IDF weights from training captions (done once, O(|train|))
        if args.flickr30k_hf:
            df_train_caps = df_caps_hf[df_caps_hf["image"].isin(set(train_ids))].reset_index(drop=True)
        else:
            df_all = load_captions_df(args.captions_csv)
            df_train_caps = df_all[df_all["image"].isin(set(train_ids))].reset_index(drop=True)
        idf = compute_idf_weights(df_train_caps)

        # Reference captions per training image
        refs_by_image: dict = {}
        for img_id in train_ids:
            refs_by_image[img_id] = df_train_caps[df_train_caps["image"] == img_id]["caption"].tolist()

        # Separate low-LR optimizer for SCST
        scst_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.scst_lr)

        if args.flickr30k_hf:
            scst_loader = get_scst_loader_hf(
                hf_ds, vocab,
                batch_size=args.scst_batch_size,
                num_workers=args.num_workers,
            )
        else:
            scst_loader = get_scst_loader(
                images_dir=args.images_dir,
                captions_csv=args.captions_csv,
                vocab=vocab,
                train_ids=train_ids,
                batch_size=args.scst_batch_size,
                num_workers=args.num_workers,
            )
        print(f"[SCST] {len(scst_loader)} batches/epoch  lr={args.scst_lr}  batch_size={args.scst_batch_size}")

        best_scst_reward = float("-inf")
        for scst_ep in range(1, args.scst_epochs + 1):
            t0 = time.time()
            mean_reward = scst_epoch(
                encoder, decoder, scst_loader, scst_optimizer,
                vocab, refs_by_image, idf, device, args,
                use_wandb=use_wandb, epoch=scst_ep,
            )
            elapsed = time.time() - t0
            print(f"== SCST epoch {scst_ep}/{args.scst_epochs}  mean_reward={mean_reward:.4f}  ({elapsed:.0f}s)")
            if use_wandb:
                wandb.log({"scst/epoch_reward": mean_reward, "scst/epoch": scst_ep})

            scst_save = {
                "epoch": scst_ep,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "vocab_size": len(vocab),
                "args": vars(args),
                "scst_reward": mean_reward,
            }
            out = Path(args.scst_checkpoints_dir) / f"ckpt_scst_epoch{scst_ep}.pt"
            safe_save(scst_save, out)
            print(f"[SCST ckpt] saved {out}")

            if mean_reward > best_scst_reward:
                best_scst_reward = mean_reward
                safe_save(scst_save, Path(args.scst_checkpoints_dir) / "ckpt_best.pt")
                print(f"[SCST] new best reward={best_scst_reward:.4f} -> saved ckpt_best.pt")

        # BLEU evaluation uses the best SCST checkpoint
        best_ckpt_path = Path(args.scst_checkpoints_dir) / "ckpt_best.pt"
        print(f"[SCST] Fine-tuning done. BLEU eval will use {best_ckpt_path}")

    # --- BLEU + METEOR evaluation on test set (millor checkpoint) ---
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    print("\n[bleu+meteor] avaluant sobre el conjunt de test...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    encoder.load_state_dict(best_ckpt["encoder"])  # carrega pesos del millor model
    decoder.load_state_dict(best_ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    if args.flickr30k_hf:
        df_caps = df_caps_hf
    else:
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
        if args.flickr30k_hf:
            hyp = simple_tokenize(caption_pil_image(test_pil[img], encoder, decoder, vocab, device))
        else:
            hyp = simple_tokenize(caption_image(f"{args.images_dir}/{img}", encoder, decoder, vocab, device))
        b1 = sentence_bleu(refs, hyp, weights=(1,0,0,0), smoothing_function=smooth)
        b4 = sentence_bleu(refs, hyp, weights=(.25,.25,.25,.25), smoothing_function=smooth)
        m  = meteor_score(refs, hyp)
        all_refs.append(refs)
        all_hyps.append(hyp)
        all_meteors.append(m)
        print(f"{img:<35} {b1:>7.3f} {b4:>7.3f} {m:>7.3f}  {' '.join(hyp)}")
        if bleu_table is not None and len(bleu_table.data) < TABLE_LIMIT:
            ref_str = " | ".join([" ".join(r) for r in refs])
            if args.flickr30k_hf:
                bleu_table.add_data(str(img), " ".join(hyp), ref_str, round(b1, 3), round(b4, 3), round(m, 3))
            else:
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
