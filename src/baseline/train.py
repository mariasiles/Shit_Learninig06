"""Training script for the Flickr8k Image Captioning baseline.

Usage:
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
import torch.nn as nn # aquí s'utilitza per crear la loss nn.CrossEntropyLoss()
from torch.nn.utils.rnn import pack_padded_sequence # per treballar amb seqüències de longitud variable.
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction  # mètriques BLEU
from nltk.translate.meteor_score import meteor_score  # mètrica METEOR (té en compte sinònims)

from src.shared.dataset import get_loaders, get_loaders_hf, split_image_ids, load_captions_df # els dataloaders i la funció per dividir el dataset
from src.baseline.model import DecoderRNN, EncoderCNN # les dues xarxes principals.
from src.baseline.sample import caption_image, caption_pil_image  # per generar captions durant l'avaluació BLEU
from src.shared.vocabulary import Vocabulary, build_vocab, build_vocab_hf, simple_tokenize, load_glove_weights, load_word2vec_weights # vocabulari, tokenitzador, GloVe i Word2Vec
from src.shared.losses import SemanticCrossEntropyLoss, build_soft_labels  # loss semàntica


def parse_args(): # a llegir tots els arguments
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="dataset/Images") # directori imatges
    p.add_argument("--captions-csv", default="dataset/captions.txt") # path fitxer captions
    p.add_argument("--vocab-path", default="dataset/vocab.pkl") # path on es guarda o carrega el vocab
    p.add_argument("--checkpoints-dir", default="checkpoints") # directori on es guarden els checkpoints de l'entrenament (epoch, vocab_size, args...) i models entrenats
    p.add_argument("--vocab-threshold", type=int, default=5) # mínim de vegades ha d'aparèixer una paraula per entrar al vocabulari (default 5, si no entra --> <unk>)

    p.add_argument("--embed-size", type=int, default=256) # mida del vector de l'embedding de les imatges i paraules
    p.add_argument("--hidden-size", type=int, default=512) # mida de l'estat ocult de la LSTM 
    p.add_argument("--num-layers", type=int, default=1) # nombre de capes apilades de la LSTM (profunditat)
    p.add_argument("--dropout", type=float, default=0.5) # probabilitat de dropout a la LSTM (regularització que apaga neurones aleatòriament durant l'entrenament per evitar overfitting)
    p.add_argument("--backbone", default="resnet152") # quina CNN preentrenada utilitzar com a encoder (resnet50 o resnet152)

    p.add_argument("--epochs", type=int, default=5) # numero de passades completes del train
    p.add_argument("--patience", type=int, default=5) # nombre d'epochs que esperarem sense millora en la val_loss abans de parar l'entrenament (early stopping)
    p.add_argument("--batch-size", type=int, default=32) # quantes mostres entrenen el model a cada pas 
    p.add_argument("--num-workers", type=int, default=2) # quants processos paral·lels carregaran dades
    p.add_argument("--lr", type=float, default=1e-3) # de l'optimitzador Adam (la mida del pas d'actualització dels pesos)
    p.add_argument("--log-step", type=int, default=20) # cada quants batches s'imprimeixen mètriques (loss, perplexity)

    p.add_argument("--glove-path", default=None,
                   help="Ruta al fitxer GloVe (ex: glove.6B.300d.txt). Si no s'especifica, usa embeddings aleatoris.")
    # GloVe: https://nlp.stanford.edu/data/glove.6B.zip → descomprimeix i passa el path
    p.add_argument("--word2vec-path", default=None,
                   help="Ruta al fitxer Word2Vec (ex: GoogleNews-vectors-negative300.bin o .txt). "
                        "S'ignora si --glove-path també s'especifica.")
    p.add_argument("--word2vec-binary", action="store_true",
                   help="Indica que el fitxer Word2Vec és en format binari (.bin). "
                        "Si no s'activa, es detecta automàticament per l'extensió.")
    p.add_argument("--freeze-embeddings", action="store_true",
                   help="Si s'activa, els pesos de l'embedding (GloVe o Word2Vec) no s'actualitzen durant l'entrenament.")
    p.add_argument("--semantic-temp", type=float, default=10.0,
                   help="Temperatura pels soft labels semàntics (amb --glove-path o --word2vec-path). Més alt = més concentrat al target.")
    # --glove-path sense --freeze-embeddings → GloVe fine-tuned (pesos GloVe + s'actualitzen)
    # --glove-path amb    --freeze-embeddings → GloVe frozen   (pesos GloVe fixos)
    # sense --glove-path                      → scratch        (pesos aleatoris, comportament original)

    # ── Flickr30k HuggingFace ──────────────────────────────────────────────
    p.add_argument("--flickr30k-hf", action="store_true",
                   help="Usa el dataset Flickr30k de HuggingFace (nlphuji/flickr30k) en lloc del CSV.")
    p.add_argument("--flickr30k-hf-cache", default="dataset/flickr30k_hf",
                   help="Carpeta cache del dataset HuggingFace.")

    p.add_argument("--wandb", action="store_true") # argument que activa wandb
    p.add_argument("--wandb-project", default="image-captioning") # nom del projecte a wandb
    p.add_argument("--wandb-entity", default=None) # nom de l'entitat (usuari o organització) a wandb, si es deixa None s'utilitzarà l'entitat per defecte de l'usuari
    p.add_argument("--run-name", default=None) # nom l'execució concreta, si es deixa None s'utilitzarà un nom generat automàticament basat en la data i hora actual
    return p.parse_args()


def get_or_build_vocab(args) -> Vocabulary: # funció que retorna l'objecte Vocabulary
    vp = Path(args.vocab_path) # ruta del vocab
    if vp.exists(): # si ja existeix el fitxer del vocab, el carrega i el retorna
        with open(vp, "rb") as f:
            return pickle.load(f)
    vocab = build_vocab(args.captions_csv, threshold=args.vocab_threshold) # si no existeix, el construeix a partir de les captions i threshold
    vp.parent.mkdir(parents=True, exist_ok=True)
    with open(vp, "wb") as f:
        pickle.dump(vocab, f) # el guarda 
    print(f"[vocab] built and saved to {vp} (size={len(vocab)})")
    return vocab # i el retorna


@torch.no_grad() # evaluate només crida el model no entrena. No calcula gradients.
def evaluate(encoder, decoder, loader, criterion, device) -> float:
    # encoder: CNN, decoder: LSTM, loader: val_loader, criterion: CrossEntropyLoss, device: cpu o gpu
    encoder.eval() # en mode evaluació desactivem dropout.
    decoder.eval()
    losses = [] # per guardar la loss de cada batch de validació
    for images, captions, lengths in loader: # recorre tots els batches del loader
        # images --> [B, 3, 224, 224], captions --> [B, T], lengths --> [B]
        images = images.to(device) # mou imatges i captions a gpu o cpu
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True).data # converteix les captions amb padding en una seqüència de tokens sense paddings, que és el format que espera la funció de pèrdua. Per exemple, si tenim captions de longituds [5, 3] i max_len=5, les captions podrien ser [[1,34,56,2,0],[1,3,2,0,0]] (on 0 és el padding). pack_padded_sequence convertirà això en una seqüència concatenada de tokens [1,34,56,2,1,3,2] i una llista de longituds [5, 3] per indicar on acaben les captions reals.
        features = encoder(images) # passa les imatges per l'encoder i obté els vectors de característiques [B, embed_size]
        outputs = decoder(features, captions, lengths) # passa les features i les captions al decoder. outputs són les prediccions de paraules per cada pas de la seqüència, amb forma [sum(lengths), vocab_size] (totes les prediccions concatenades sense padding)
        loss = criterion(outputs, targets) # calcula la CrossEntropyLoss comparant outputs amb targets
        losses.append(loss.item()) # afageix la loss del batch a la llista
    return float(np.mean(losses)) # retorna la mitjana de totes les losses de validació


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # entrena amb gpu si està disponible, si no cpu
    print(f"[device] {device}") # imprimeix quin s'utilitzarà

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True) # crea la carpeta de checkpoints si no existeix

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
        vocab = get_or_build_vocab(args) # carrega o construeix el vocabulari
        train_loader, val_loader, _, _ = get_loaders(
            images_dir=args.images_dir,
            captions_csv=args.captions_csv,
            vocab=vocab,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ) # crea els data loaders

    print(f"[vocab] size = {len(vocab)}") # mostra mida del vocabulari
    print(f"[data] train batches={len(train_loader)}  val batches={len(val_loader)}") # mira quants batches hi ha al train i a la validació

    # --- Embeddings: scratch / GloVe / Word2Vec (fine-tuned o frozen) ---
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
    # --------------------------------------------------------------

    encoder = EncoderCNN(args.embed_size, backbone=args.backbone).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers, dropout=args.dropout,
                         pretrained_weights=pretrained_weights,
                         freeze_embeddings=args.freeze_embeddings).to(device)

    if pretrained_weights is not None:
        # Amb GloVe o Word2Vec: soft labels semàntics (paraules similars penalitzades menys)
        soft_lbls = build_soft_labels(decoder.embed.weight.data.cpu(), temperature=args.semantic_temp)
        criterion = SemanticCrossEntropyLoss(soft_lbls).to(device)
        print(f"[loss] SemanticCrossEntropy (temp={args.semantic_temp}) — soft labels des de {emb_type}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[loss] CrossEntropyLoss estàndard (sense embeddings preentrenats)")
    params = ( # llista de paràmetres que entrenarem
        list(decoder.parameters()) # inclou tots els paràmetres del decoder
        + list(encoder.linear.parameters()) # inclou els de l'encoder (només s'entrenaven les capes linear i bn, les altres es congelaven)
        + list(encoder.bn.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=args.lr) # crea l'optimitzador Adam. Només actualitzarà els paràmetres especificats a params, i li especifiquem la learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    ) # redueix el LR a la meitat si la val_loss no millora durant 2 epochs

    use_wandb = args.wandb 
    if use_wandb: # si hem activat wandb
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, config=vars(args)) # li passem els arguments
        wandb.config.update({"vocab_size": len(vocab), "embedding_type": emb_type}) # afegeix mida del vocabulari i tipus d'embedding a wandb

    train_losses: list[float] = [] # per guardar les losses de l'entrenament
    val_losses: list[float] = [] # losses de la val (de cada epoch)

    best_val_loss = float("inf") # inicialitza la millor loss com infinit 
    patience_counter = 0 # contador de l'early stopping
    global_step = 0 # comptador de batches processats
    for epoch in range(1, args.epochs + 1): # bucle d'epochs
        encoder.train() # posa l'encoder i el decoder en mode entrenament
        decoder.train() 
        t0 = time.time() # guarda el temps inicial de la epoch
        for i, (images, captions, lengths) in enumerate(train_loader): # recorre tots els batches d'entrenament; i --> index del batch, els altres són les dades del batch
            images = images.to(device, non_blocking=True) # mou les imatges a gpu o cpu. non_blocking pot accelerar la transferència si el DataLoader utilitza memòria pinned (que ho fa)
            captions = captions.to(device, non_blocking=True) # mou les captions al dispositiu
            targets = pack_padded_sequence(captions, lengths, batch_first=True).data # crea els objectius reals del model eliminant padding (perquè el model aprengui a predir <pad>)

            features = encoder(images) # passa les imatges per la CNN
            outputs = decoder(features, captions, lengths) # passa els embeddings de les imgs, les captions i les lenghts a la LSTM
            # per cada token dona una distribució sobre el vocabulari
            loss = criterion(outputs, targets) # calcula la Cross Entropy Loss comparant les prediccions del model amb les paraules correctes

            # BACKPROPAGATION
            optimizer.zero_grad() # posa els gradients a zero (perquè no s'acumulin amb els antics)
            loss.backward() # calcula els gradients
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step() # actualitza els pesos --> APRÈN yuppi

            global_step += 1 # sumem 1 al comptador de batchos
            train_losses.append(loss.item()) # guarda la loss del batch
            if i % args.log_step == 0: # comprova si toca imprimir la informació
                ppl = float(np.exp(min(loss.item(), 20))) # calcula la perplexity (com més baixa millor)
                print(f"epoch {epoch}/{args.epochs}  step {i}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  ppl={ppl:.2f}") # imprimeix info de l'entrenament
                if use_wandb: # si wandb està activat registra mètriques
                    wandb.log({"train/loss": loss.item(), "train/perplexity": ppl,
                               "epoch": epoch, "step": global_step})

        val_loss = evaluate(encoder, decoder, val_loader, criterion, device) # validació després de cada epoch, que és la loss mirjana de validació
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        val_ppl = float(np.exp(min(val_loss, 20))) # perplexity de la validació
        elapsed = time.time() - t0 # es tanca el temps per saber quan ha durat la epoch
        print(f"== epoch {epoch} done  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  ({elapsed:.0f}s)")
        if use_wandb:
            wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "epoch": epoch,
                       "lr": optimizer.param_groups[0]["lr"]}) # registra metriques a wandb

        ckpt = { # diccionari amb la info que es vol guardar
            "epoch": epoch, # en quina epoch s'ha guardat el checkpoin
            "encoder": encoder.state_dict(), # guarda pesos de l'encoder
            "decoder": decoder.state_dict(), # del decoder
            "vocab_size": len(vocab), 
            "args": vars(args),
        }
        out = Path(args.checkpoints_dir) / f"ckpt_epoch{epoch}.pt" # ruta del checkpoint d'aquesta epoch
        torch.save(ckpt, out) # guarda al disc
        print(f"[ckpt] saved {out}")

        if val_loss < best_val_loss: # comprova si la validació ha millorat
            best_val_loss = val_loss # actualitza la millor loss
            patience_counter = 0 # reinicia comptador de paciència
            best_out = Path(args.checkpoints_dir) / "ckpt_best.pt" # defineix la ruta del millor model
            torch.save(ckpt, best_out) # guarda
            print(f"[early_stop] new best val_loss={best_val_loss:.4f} -> saved ckpt_best.pt") # mostra que hi ha un nou millor model
        else: # si validació no ha millorat
            patience_counter += 1 # incrementa comptador de paciència
            print(f"[early_stop] no improvement ({patience_counter}/{args.patience})") # mostra quantes epochs seguides sense millorar
            if patience_counter >= args.patience: # si hem arribat al límit de paciència
                print(f"[early_stop] patience exhausted, stopping at epoch {epoch}") # mostra que l'entrenament s'atura
                break

    # --- loss curve ---
    steps_per_epoch = len(train_loader) # quarda quants batches té una epoch
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)) # figura amb 2 gràfiques en una fila

    axes[0].plot(train_losses, alpha=0.6, label="train (per batch)") # la de le'squerra és la gràfica d'entrenament
    for e in range(1, args.epochs + 1):
        axes[0].axvline(e * steps_per_epoch, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("batch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Train loss")
    axes[0].legend()

    axes[1].plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="val") # la de la dreta és la gràfica de validació
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy loss")
    axes[1].set_title("Val loss per epoch")
    axes[1].legend()

    plt.tight_layout()
    plot_path = Path(args.checkpoints_dir) / "loss_curve.png" # on es guarda la imatge
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[plot] saved {plot_path}")
    # ------------------

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

    if args.flickr30k_hf:
        df_caps = df_caps_hf
    else:
        _, _, test_ids = split_image_ids(args.captions_csv)  # agafa els IDs del test set
        df_caps = load_captions_df(args.captions_csv)  # llegeix totes les captions (Flickr8k o Flickr30k)
    smooth = SmoothingFunction().method1  # suavitzat per evitar BLEU-4 = 0

    all_refs, all_hyps = [], []
    all_meteors = []
    bleu_table = wandb.Table(columns=["image", "generated_caption", "reference_captions", "BLEU-1", "BLEU-4", "METEOR"]) if use_wandb else None
    images_dir_abs = Path(args.images_dir).resolve()
    TABLE_LIMIT = 200  # WandB no renderitza bé les imatges amb >200 files

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
                from PIL import Image as PILImage
                pil_img = PILImage.open(str(images_dir_abs / img)).convert("RGB")
                bleu_table.add_data(wandb.Image(pil_img), " ".join(hyp), ref_str, round(b1, 3), round(b4, 3), round(m, 3))

    cb1 = corpus_bleu(all_refs, all_hyps, weights=(1,0,0,0))
    cb4 = corpus_bleu(all_refs, all_hyps, weights=(.25,.25,.25,.25))
    cm  = float(np.mean(all_meteors))
    print("-" * 110)
    print(f"[bleu] Corpus BLEU-1: {cb1:.3f}  BLEU-4: {cb4:.3f}  METEOR: {cm:.3f}")

    if use_wandb:
        wandb.log({"bleu_eval_table": bleu_table, "bleu/corpus_bleu1": cb1, "bleu/corpus_bleu4": cb4, "bleu/meteor": cm})
    # ------------------------------------------------

    if use_wandb:
        wandb.finish() # tanca corectament la run de wandb perquè marqui l'experiment com acabat


if __name__ == "__main__":
    main()
