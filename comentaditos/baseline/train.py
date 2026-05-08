"""
train.py  —  Script d'entrenament complet (src/baseline/train.py)
==========================================================
Versió base que implementa el pipeline d'entrenament estàndard (Encoder CNN + Decoder LSTM)
per a la generació de descripcions d'imatges.

Ús des del terminal:
    python -m src.train --epochs 5 --batch-size 32 --wandb
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
import torch.nn as nn # per crear la loss nn.CrossEntropyLoss()
from torch.nn.utils.rnn import pack_padded_sequence # per treballar amb seqüències de longitud variable
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction  # mètriques BLEU
from nltk.translate.meteor_score import meteor_score  # mètrica METEOR (té en compte sinònims)

from src.shared.dataset import get_loaders, get_loaders_hf, split_image_ids, load_captions_df 
from src.baseline.model import DecoderRNN, EncoderCNN # les dues xarxes principals (Baseline)
from src.baseline.sample import caption_image, caption_pil_image  # per generar captions en l'avaluació
from src.shared.vocabulary import Vocabulary, build_vocab, build_vocab_hf, simple_tokenize, load_glove_weights, load_word2vec_weights 
from src.shared.losses import SemanticCrossEntropyLoss, build_soft_labels  # loss semàntica avançada


# ═════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    """Defineix tots els hiperparàmetres configurables des del terminal."""
    p = argparse.ArgumentParser()
    
    # ── Rutes i Fitxers ──────────────────────────────────────────────────────
    p.add_argument("--images-dir", default="dataset/Images") # directori imatges
    p.add_argument("--captions-csv", default="dataset/captions.txt") # path fitxer captions
    p.add_argument("--vocab-path", default="dataset/vocab.pkl") # on es guarda/carrega el vocab
    p.add_argument("--checkpoints-dir", default="checkpoints") # on es guarden els pesos entrenats
    p.add_argument("--vocab-threshold", type=int, default=5) # freqüència mínima per entrar al vocabulari

    # ── Arquitectura del Model ───────────────────────────────────────────────
    p.add_argument("--embed-size", type=int, default=256) # dimensió de l'espai d'embedding
    p.add_argument("--hidden-size", type=int, default=512) # dimensió de l'estat ocult de la LSTM 
    p.add_argument("--num-layers", type=int, default=1) # profunditat de la LSTM
    p.add_argument("--dropout", type=float, default=0.5) # regularització per evitar overfitting
    p.add_argument("--backbone", default="resnet152") # CNN preentrenada (resnet50/152)

    # ── Paràmetres d'Entrenament ──────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=5) # passades completes pel dataset
    p.add_argument("--patience", type=int, default=5) # epochs d'espera per a l'Early Stopping
    p.add_argument("--batch-size", type=int, default=32) # mostres per iteració
    p.add_argument("--num-workers", type=int, default=2) # paral·lelisme en la càrrega de dades
    p.add_argument("--lr", type=float, default=1e-3) # learning rate per a l'optimitzador Adam
    p.add_argument("--log-step", type=int, default=20) # freqüència de logs al terminal

    # ── Embeddings Preentrenats ──────────────────────────────────────────────
    p.add_argument("--glove-path", default=None,
                   help="Ruta al fitxer GloVe (ex: glove.6B.300d.txt).")
    p.add_argument("--word2vec-path", default=None,
                   help="Ruta al fitxer Word2Vec. S'ignora si s'usa GloVe.")
    p.add_argument("--word2vec-binary", action="store_true",
                   help="Indica format binari (.bin).")
    p.add_argument("--freeze-embeddings", action="store_true",
                   help="Si s'activa, els pesos de l'embedding no s'actualitzen (frozen).")
    p.add_argument("--semantic-temp", type=float, default=10.0,
                   help="Temperatura pels soft labels semàntics.")

    # ── Dataset Flickr30k (HuggingFace) ──────────────────────────────────────
    p.add_argument("--flickr30k-hf", action="store_true",
                   help="Usa la versió de HuggingFace en lloc del CSV local.")
    p.add_argument("--flickr30k-hf-cache", default="dataset/flickr30k_hf")

    # ── Mètriques i Monitoring (WandB) ────────────────────────────────────────
    p.add_argument("--wandb", action="store_true") # activa el registre a Weights & Biases
    p.add_argument("--wandb-project", default="image-captioning")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--run-name", default=None)
    
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
# UTILITATS
# ═════════════════════════════════════════════════════════════════════

def get_or_build_vocab(args) -> Vocabulary:
    """Gestiona el vocabulari: el carrega de disc o el construeix de zero."""
    vp = Path(args.vocab_path)
    if vp.exists():
        with open(vp, "rb") as f:
            return pickle.load(f)
    
    # Construcció des de les captions si no existeix el fitxer .pkl
    vocab = build_vocab(args.captions_csv, threshold=args.vocab_threshold)
    vp.parent.mkdir(parents=True, exist_ok=True)
    with open(vp, "wb") as f:
        pickle.dump(vocab, f)
    print(f"[vocab] built and saved to {vp} (size={len(vocab)})")
    return vocab


@torch.no_grad()
def evaluate(encoder, decoder, loader, criterion, device) -> float:
    """Calcula la pèrdua (loss) mitjana en el conjunt de validació.
    
    Diferències respecte a train:
        S'usa @torch.no_grad() per estalviar memòria i temps, ja que no 
        calen els gradients per a l'avaluació.
    """
    encoder.eval() # Desactiva capes com Dropout o Batch Normalization
    decoder.eval()
    losses = []
    
    for images, captions, lengths in loader:
        images = images.to(device)
        captions = captions.to(device)
        
        # Formatat de les dades per a la Loss (CrossEntropy)
        targets = pack_padded_sequence(captions, lengths, batch_first=True).data
        
        # Forward pass
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        
    return float(np.mean(losses))


# ═════════════════════════════════════════════════════════════════════
# BUCLE PRINCIPAL D'ENTRENAMENT
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # ── Preparació de Dades ───────────────────────────────────────────────────
    if args.flickr30k_hf:
        # Pipeline per al dataset des de HuggingFace
        from datasets import load_dataset
        print("[data] carregant Flickr30k HuggingFace...")
        hf_ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True,
                             cache_dir=args.flickr30k_hf_cache)
        
        # Gestió del vocabulari específica per a HF
        vp = Path(args.vocab_path)
        if vp.exists():
            with open(vp, "rb") as f:
                import pickle; vocab = pickle.load(f)
        else:
            vocab = build_vocab_hf(hf_ds, threshold=args.vocab_threshold)
            with open(vp, "wb") as f:
                import pickle; pickle.dump(vocab, f)

        train_loader, val_loader, _ = get_loaders_hf(
            hf_ds, vocab, batch_size=args.batch_size, num_workers=args.num_workers)

        # Preparació de referències per a l'avaluació final
        full = hf_ds["test"]
        test_rows = full.filter(lambda x: x["split"] == "test")
        test_ids  = [r["filename"] for r in test_rows]
        records = []
        for r in full:
            for cap in r["caption"]:
                records.append({"image": r["filename"], "caption": cap})
        import pandas as _pd
        df_caps_hf = _pd.DataFrame(records)
        test_pil = {r["filename"]: r["image"] for r in test_rows}
    else:
        # Pipeline estàndard amb CSV local
        vocab = get_or_build_vocab(args)
        train_loader, val_loader, _, _ = get_loaders(
            images_dir=args.images_dir,
            captions_csv=args.captions_csv,
            vocab=vocab,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    print(f"[vocab] size = {len(vocab)}")
    print(f"[data] train batches={len(train_loader)}  val batches={len(val_loader)}")

    # ── Gestió d'Embeddings ──────────────────────────────────────────────────
    # Es permet triar entre entrenar de zero (scratch) o usar vectors preentrenats
    pretrained_weights = None
    if args.glove_path:
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

    # ── Inicialització de Models ─────────────────────────────────────────────
    encoder = EncoderCNN(args.embed_size, backbone=args.backbone).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers, dropout=args.dropout,
                         pretrained_weights=pretrained_weights,
                         freeze_embeddings=args.freeze_embeddings).to(device)

    # ── Configuració de la Loss Semàntica ─────────────────────────────────────
    # Si usem GloVe/Word2Vec, podem aplicar una loss que penalitzi menys els sinònims
    if pretrained_weights is not None:
        soft_lbls = build_soft_labels(decoder.embed.weight.data.cpu(), temperature=args.semantic_temp)
        criterion = SemanticCrossEntropyLoss(soft_lbls).to(device)
        print(f"[loss] SemanticCrossEntropy (temp={args.semantic_temp})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[loss] CrossEntropyLoss estàndard")

    # ── Optimitzador i Scheduler ──────────────────────────────────────────────
    # Entrenem tota la LSTM però només les darreres capes de la CNN (BN i FC)
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    # Reducció dinàmica del LR si la val_loss s'estanca
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # ── Monitorització WandB ──────────────────────────────────────────────────
    use_wandb = args.wandb 
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, config=vars(args))
        wandb.config.update({"vocab_size": len(vocab), "embedding_type": emb_type})

    # ── Bucle d'Èpoques ───────────────────────────────────────────────────────
    train_losses, val_losses = [], []
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
            
            # Eliminem padding per calcular la loss sobre tokens reals
            targets = pack_padded_sequence(captions, lengths, batch_first=True).data

            # Forward Pass
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)

            # Backpropagation i actualització de pesos
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping per evitar explosions del gradient en RNNs
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            # Logging
            global_step += 1
            train_losses.append(loss.item())
            if i % args.log_step == 0:
                ppl = float(np.exp(min(loss.item(), 20))) # Perplexitat: exp(loss)
                print(f"epoch {epoch}/{args.epochs}  step {i}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  ppl={ppl:.2f}")
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/perplexity": ppl,
                               "epoch": epoch, "step": global_step})

        # ── Validació i Checkpointing ────────────────────────────────────────
        val_loss = evaluate(encoder, decoder, val_loader, criterion, device)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        val_ppl = float(np.exp(min(val_loss, 20)))
        elapsed = time.time() - t0
        print(f"== epoch {epoch} done  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  ({elapsed:.0f}s)")
        
        if use_wandb:
            wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "epoch": epoch,
                       "lr": optimizer.param_groups[0]["lr"]})

        # Guardat del checkpoint actual
        ckpt = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "vocab_size": len(vocab), 
            "args": vars(args),
        }
        out = Path(args.checkpoints_dir) / f"ckpt_epoch{epoch}.pt"
        torch.save(ckpt, out)

        # Lògica d'Early Stopping i guardat del millor model (best)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_out = Path(args.checkpoints_dir) / "ckpt_best.pt"
            torch.save(ckpt, best_out)
            print(f"[early_stop] new best val_loss={best_val_loss:.4f} -> saved ckpt_best.pt")
        else:
            patience_counter += 1
            print(f"[early_stop] no improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"[early_stop] patience exhausted, stopping at epoch {epoch}")
                break

    # ── Generació de Gràfiques ───────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gràfica de Train Loss
    axes[0].plot(train_losses, alpha=0.6, label="train (per batch)")
    for e in range(1, args.epochs + 1):
        axes[0].axvline(e * steps_per_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Train loss")
    axes[0].legend()

    # Gràfica de Val Loss
    axes[1].plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="val")
    axes[1].set_title("Val loss per epoch")
    axes[1].legend()

    plt.tight_layout()
    plot_path = Path(args.checkpoints_dir) / "loss_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[plot] saved {plot_path}")

    # ── Avaluació Final (BLEU + METEOR) ──────────────────────────────────────
    # Carreguem el millor model guardat per avaluar el conjunt de test
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    
    print("\n[bleu+meteor] avaluant sobre el conjunt de test...")
    best_ckpt = torch.load(Path(args.checkpoints_dir) / "ckpt_best.pt", map_location=device)
    encoder.load_state_dict(best_ckpt["encoder"]) 
    decoder.load_state_dict(best_ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    # Preparació d'IDs i referències segons el dataset usat
    if args.flickr30k_hf:
        df_caps = df_caps_hf
    else:
        _, _, test_ids = split_image_ids(args.captions_csv) 
        df_caps = load_captions_df(args.captions_csv) 
    
    smooth = SmoothingFunction().method1 # Evita que BLEU-4 sigui 0 si no hi ha 4-grams
    all_refs, all_hyps, all_meteors = [], [], []
    
    # Taula de WandB per visualitzar resultats qualitatius
    bleu_table = wandb.Table(columns=["image", "generated_caption", "reference_captions", "BLEU-1", "BLEU-4", "METEOR"]) if use_wandb else None
    TABLE_LIMIT = 200 

    print(f"Evaluating {len(test_ids)} test images...")
    print(f"{'Image':<35} {'BLEU-1':>7} {'BLEU-4':>7} {'METEOR':>7}  Caption")
    print("-" * 110)

    for img in test_ids:
        # Referències (tokenitzades)
        refs = [simple_tokenize(c) for c in df_caps[df_caps["image"] == img]["caption"].tolist()]
        
        # Hipòtesi (generació greedy del model)
        if args.flickr30k_hf:
            hyp = simple_tokenize(caption_pil_image(test_pil[img], encoder, decoder, vocab, device))
        else:
            hyp = simple_tokenize(caption_image(f"{args.images_dir}/{img}", encoder, decoder, vocab, device))
        
        # Càlcul de mètriques per frase
        b1 = sentence_bleu(refs, hyp, weights=(1,0,0,0), smoothing_function=smooth)
        b4 = sentence_bleu(refs, hyp, weights=(.25,.25,.25,.25), smoothing_function=smooth)
        m  = meteor_score(refs, hyp)
        
        all_refs.append(refs)
        all_hyps.append(hyp)
        all_meteors.append(m)
        
        print(f"{img:<35} {b1:>7.3f} {b4:>7.3f} {m:>7.3f}  {' '.join(hyp)}")
        
        # Log a la taula de WandB
        if bleu_table is not None and len(bleu_table.data) < TABLE_LIMIT:
            ref_str = " | ".join([" ".join(r) for r in refs])
            if args.flickr30k_hf:
                bleu_table.add_data(str(img), " ".join(hyp), ref_str, round(b1, 3), round(b4, 3), round(m, 3))
            else:
                from PIL import Image as PILImage
                images_dir_abs = Path(args.images_dir).resolve()
                pil_img = PILImage.open(str(images_dir_abs / img)).convert("RGB")
                bleu_table.add_data(wandb.Image(pil_img), " ".join(hyp), ref_str, round(b1, 3), round(b4, 3), round(m, 3))

    # Mètriques de Corpus (global sobre tot el test set)
    cb1 = corpus_bleu(all_refs, all_hyps, weights=(1,0,0,0))
    cb4 = corpus_bleu(all_refs, all_hyps, weights=(.25,.25,.25,.25))
    cm  = float(np.mean(all_meteors))
    
    print("-" * 110)
    print(f"[bleu] Corpus BLEU-1: {cb1:.3f}  BLEU-4: {cb4:.3f}  METEOR: {cm:.3f}")

    if use_wandb:
        wandb.log({"bleu_eval_table": bleu_table, "bleu/corpus_bleu1": cb1, "bleu/corpus_bleu4": cb4, "bleu/meteor": cm})
        wandb.finish()


if __name__ == "__main__":
    main()