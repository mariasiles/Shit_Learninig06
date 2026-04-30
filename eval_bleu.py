"""BLEU evaluation on the Flickr8k test set, optionally logged to W&B.

Example:
    python eval_bleu.py \
        --checkpoint checkpoints/ckpt_best.pt \
        --wandb \
        --wandb-project image-captioning \
        --wandb-entity learning6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction # importa les funcions de BLEU

# This lets the file work both when it is executed from the project root
# and when it is executed as a module from a src/ package.

from dataset import split_image_ids # divideix les imatges en train, val i test
from sample import caption_image, load_checkpoint # genera una caption per una imarge concreta 
from vocabulary import simple_tokenize # converteix una frase en una llista deparaules/tokens 



def parse_args() -> argparse.Namespace: # llegim els arguments de terminal
    p = argparse.ArgumentParser(description="Evaluate an image-captioning checkpoint with BLEU.")
    p.add_argument("--checkpoint", default="checkpoints/ckpt_best.pt", 
                   help="Path to the checkpoint to evaluate.") # ruta del model entrenat a evaluar
    p.add_argument("--vocab-path", default="data/flickr8k/vocab.pkl",
                   help="Path to the vocabulary pickle used by the checkpoint.") # ruta del vocabulari guardat
    p.add_argument("--images-dir", default="data/flickr8k/Images",
                   help="Directory containing the Flickr8k images.") # carpeta on hi ha les imarges
    p.add_argument("--captions-csv", default="data/flickr8k/captions.txt",
                   help="CSV file with columns image and caption.") # fitxer on hi ha les captions
    p.add_argument("--max-images", type=int, default=None,
                   help="Evaluate only the first N test images. Useful for a quick smoke test.") # per només avaluar unes quantes imarges
    p.add_argument("--no-wandb-table-images", action="store_true",
                   help="Log image filenames instead of wandb.Image objects. Faster and lighter.") # si s'activa això no pujem la imatge com a objecte visual, només el nom de la foto.

    p.add_argument("--wandb", action="store_true", # activa wandb
                   help="Log BLEU metrics and the prediction table to W&B.")
    p.add_argument("--wandb-project", default="image-captioning", # nom del projecte
                   help="W&B project name.")
    p.add_argument("--wandb-entity", default="learning6", # nom de la entitat (la nostra creada per aquest projecte)
                   help="W&B entity/team/user name.")
    p.add_argument("--run-name", default="bleu-eval-fulltest", # nom de la execució
                   help="Name for the W&B evaluation run.")
    return p.parse_args()


def main() -> None:
    args = parse_args() # processa els arguments passats

    checkpoint = Path(args.checkpoint)
    vocab_path = Path(args.vocab_path)
    images_dir = Path(args.images_dir)
    captions_csv = Path(args.captions_csv)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Train first so ckpt_best.pt exists."
        )
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not captions_csv.exists():
        raise FileNotFoundError(f"Captions CSV not found: {captions_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # assigna device si pot a gpu si no a cpu
    print(f"[device] {device}")
    print(f"[checkpoint] {checkpoint}")

    encoder, decoder, vocab = load_checkpoint(str(checkpoint), str(vocab_path), device) # crea el model entrenat
    _, _, test_ids = split_image_ids(str(captions_csv)) # particions del dataset (ens quedem amb test)
    if args.max_images is not None:
        test_ids = test_ids[: args.max_images] # limitem el numero d'imatges si cal

    df = pd.read_csv(captions_csv) # llegeix captions reals
    smooth = SmoothingFunction().method1 # una correcció per evitar que BLEU-4 doni 0 massa sovint (BLEU-4 mira coincidències de grups de 4 paraules)

    run = None # variables que es queden a None si no inicialitzem wandb
    table = None
    if args.wandb:
        import wandb

        run = wandb.init( # inicialitzes la run de wandb
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.run_name,
            config={
                "checkpoint": str(checkpoint),
                "vocab_path": str(vocab_path),
                "images_dir": str(images_dir),
                "captions_csv": str(captions_csv),
                "n_images": len(test_ids),
                "max_images": args.max_images,
            },
        )
        table = wandb.Table(columns=[ # crees la taula d'avaluació
            "image", "generated_caption", "reference_captions", "BLEU-1", "BLEU-4"
        ])

    all_refs, all_hyps = [], []

    print(f"Evaluating {len(test_ids)} test images...")
    print(f"{'Image':<35} {'BLEU-1':>7} {'BLEU-4':>7}  Caption")
    print("-" * 100)

    for img in test_ids: # per cada imarge
        img_path = images_dir / img # reconstruim path imatge
        refs = [simple_tokenize(c) for c in df[df["image"] == img]["caption"].tolist()] # agafa les captions reals de la imatge i les tokenitza
        generated_caption = caption_image(str(img_path), encoder, decoder, vocab, device) # genera la caption
        hyp = simple_tokenize(generated_caption) # converteix la caption a tokens

        b1 = sentence_bleu(refs, hyp, weights=(1, 0, 0, 0), smoothing_function=smooth) # calcula BLEU-1 que mira coincidències de paraules individuals (si coincideix una bleu és alt)
        b4 = sentence_bleu(refs, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) # calcla BLEU-4 que mira coincidencies de 1, 2, 3 i 4 grups de paraules

        all_refs.append(refs) # afegeix les captions reals i la predicció a les llistes globals
        all_hyps.append(hyp) # després s'utilitzen per calcular el BLEU de tot el corpus

        ref_str = " | ".join([" ".join(r) for r in refs]) # converteix les captions tokenitzades en un únic string separat per |
        caption_str = " ".join(hyp) # converteix la caption en un string
        print(f"{img:<35} {b1:>7.3f} {b4:>7.3f}  {caption_str}") # imprimeix els resultats

        if table is not None: # si estem untilitzant wandb 
            image_cell = str(img_path) if args.no_wandb_table_images else wandb.Image(str(img_path)) # si --no-wandb-table-images només guarda la ruta, si no, puja la imatge real
            table.add_data(image_cell, caption_str, ref_str, round(b1, 3), round(b4, 3)) # afeteix una fila a la taula

    cb1 = corpus_bleu(all_refs, all_hyps, weights=(1, 0, 0, 0), smoothing_function=smooth) # calcula BLEU-1 sobre tot el corpus
    cb4 = corpus_bleu(all_refs, all_hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) # i BLEU-4

    print("-" * 100)
    print(f"{'Corpus BLEU':<35} {cb1:>7.3f} {cb4:>7.3f}") # mostra el resultat final, que és el més imortant que resumeix el rendiment global del model.

    if args.wandb: # si els paràmetres estàn activat, puja la taula al wandb
        import wandb

        wandb.log({
            "bleu/corpus_bleu1": cb1,
            "bleu/corpus_bleu4": cb4,
            "eval_table": table,
        })
        print(f"\nW&B: {run.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
