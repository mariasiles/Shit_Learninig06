"""Training script for the Flickr8k Image Captioning baseline.

Usage:
    python -m src.train --epochs 5 --batch-size 32 --wandb
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn # aquí s'utilitza per crear la loss nn.CrossEntropyLoss()
from torch.nn.utils.rnn import pack_padded_sequence # per treballar amb seqüències de longitud variable.

from dataset import get_loaders # els dataloaders són objectes que donen batches amb imatges, captions i longituds.
from model import DecoderRNN, EncoderCNN # les dues xarxes principals.
from vocabulary import Vocabulary, build_vocab # construir vocabulari a partir de les captions.


def parse_args(): # a llegir tots els arguments
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="data/flickr8k/Images") # directori imatges
    p.add_argument("--captions-csv", default="data/flickr8k/captions.txt") # path fitxer captions
    p.add_argument("--vocab-path", default="data/flickr8k/vocab.pkl") # path on es guarda o carrega el vocab
    p.add_argument("--checkpoints-dir", default="checkpoints") # directori on es guarden els checkpoints de l'entrenament (epoch, vocab_size, args...) i models entrenats
    p.add_argument("--vocab-threshold", type=int, default=5) # mínim de vegades ha d'aparèixer una paraula per entrar al vocabulari (default 5, si no entra --> <unk>)

    p.add_argument("--embed-size", type=int, default=256) # mida del vector de l'embedding de les imatges i paraules
    p.add_argument("--hidden-size", type=int, default=512) # mida de l'estat ocult de la LSTM 
    p.add_argument("--num-layers", type=int, default=1) # nombre de capes apilades de la LSTM (profunditat)
    p.add_argument("--dropout", type=float, default=0.5) # probabilitat de dropout a la LSTM (regularització que apaga neurones aleatòriament durant l'entrenament per evitar overfitting)
    p.add_argument("--backbone", default="resnet50") # quina CNN preentrenada utilitzar com a encoder (resnet50 o resnet152)

    p.add_argument("--epochs", type=int, default=5) # numero de passades completes del train
    p.add_argument("--patience", type=int, default=5) # nombre d'epochs que esperarem sense millora en la val_loss abans de parar l'entrenament (early stopping)
    p.add_argument("--batch-size", type=int, default=32) # quantes mostres entrenen el model a cada pas 
    p.add_argument("--num-workers", type=int, default=2) # quants processos paral·lels carregaran dades
    p.add_argument("--lr", type=float, default=1e-3) # de l'optimitzador Adam (la mida del pas d'actualització dels pesos)
    p.add_argument("--log-step", type=int, default=20) # cada quants batches s'imprimeixen mètriques (loss, perplexity)

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
    vocab = get_or_build_vocab(args) # carrega o construeix el vocabulari
    print(f"[vocab] size = {len(vocab)}") # mostra mida del vocabulari

    train_loader, val_loader, _, _ = get_loaders(
        images_dir=args.images_dir,
        captions_csv=args.captions_csv,
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    ) # crea els data loaders
    print(f"[data] train batches={len(train_loader)}  val batches={len(val_loader)}") # mira quants batches hi ha al train i a la validació

    encoder = EncoderCNN(args.embed_size, backbone=args.backbone).to(device) # crea l'encoder i l'envia a gpu (o cpu)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers, dropout=args.dropout).to(device) # crea decoder i envia

    criterion = nn.CrossEntropyLoss() # crea la funció de pèrdua (CrossEntropyLoss quan multi-class). Les classes són paraules. si el vocabulari té 2982 paraules, el model escull entre 2982 classes.
    params = ( # llista de paràmetres que entrenarem
        list(decoder.parameters()) # inclou tots els paràmetres del decoder
        + list(encoder.linear.parameters()) # inclou els de l'encoder (només s'entrenaven les capes linear i bn, les altres es congelaven)
        + list(encoder.bn.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=args.lr) # crea l'optimitzador Adam. Només actualitzarà els paràmetres especificats a params, i li especifiquem la learning rate.

    use_wandb = args.wandb 
    if use_wandb: # si hem activat wandb
        import wandb 
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, config=vars(args)) # li passem els arguments
        wandb.config.update({"vocab_size": len(vocab)}) # afegeix també la mida del vocabulari a wandb

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
        val_ppl = float(np.exp(min(val_loss, 20))) # perplexity de la validació
        elapsed = time.time() - t0 # es tanca el temps per saber quan ha durat la epoch
        print(f"== epoch {epoch} done  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  ({elapsed:.0f}s)")
        if use_wandb:
            wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "epoch": epoch}) # registra metriques a wandb

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

    axes[1].plot(range(1, args.epochs + 1), val_losses, marker="o", label="val") # la de la dreta és la gràfica de validació
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

    if use_wandb:
        wandb.finish() # tanca corectament la run de wandb perquè marqui l'experiment com acabat


if __name__ == "__main__":
    main()
