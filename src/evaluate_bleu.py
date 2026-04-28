"""
Càlcul del BLEU Score - Image Captioning
------------------------------------------
El BLEU (Bilingual Evaluation Understudy) mesura com de semblants són
les frases generades per la xarxa respecte als peus de foto originals.

- BLEU-1: compara paraules individuals
- BLEU-2: compara parells de paraules consecutives (bigrames)
- BLEU-4: compara grups de 4 paraules (el més usat a la literatura)

Un BLEU-4 de 0.30 o més és considerat "acceptable" per Flickr8k.

Ús:
    python src/evaluate_bleu.py --checkpoint checkpoints/ckpt_epoch5.pt \
                                 --vocab data/flickr8k/vocab.pkl \
                                 --images-dir dataset/Images \
                                 --captions-csv dataset/captions.txt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Afegim el directori arrel al path per trobar els mòduls locals
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from PIL import Image
from tqdm import tqdm

try:
    from src.dataset import get_transform, split_image_ids, Flickr8kDataset
    from src.model import DecoderRNN, EncoderCNN
    from src.vocabulary import Vocabulary, simple_tokenize
except ImportError:
    from dataset import get_transform, split_image_ids, Flickr8kDataset
    from model import DecoderRNN, EncoderCNN
    from vocabulary import Vocabulary, simple_tokenize

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    # Portability: use root_dir for defaults
    p.add_argument("--checkpoint", default=None, help="Fitxer .pt del model entrenat (per defecte agafa l'últim)")
    p.add_argument("--vocab", default=str(root_dir / "dataset/vocab.pkl"), help="Fitxer vocab.pkl")
    p.add_argument("--images-dir", default=str(root_dir / "dataset/Images"))
    p.add_argument("--captions-csv", default=str(root_dir / "dataset/captions.txt"))
    p.add_argument("--num-images", type=int, default=None,
                   help="Nº d'imatges de test per avaluar (None = totes)")
    p.add_argument("--max-len", type=int, default=25,
                   help="Longitud màxima de les frases generades")
    
    args = p.parse_args()

    # Si no s'ha especificat checkpoint, busquem l'últim a la carpeta checkpoints
    if args.checkpoint is None:
        ckpt_dir = root_dir / "checkpoints"
        ckpts = sorted(list(ckpt_dir.glob("*.pt")), key=lambda x: x.stat().st_mtime)
        if ckpts:
            args.checkpoint = str(ckpts[-1])
            print(f"[auto] Usant l'últim checkpoint trobat: {args.checkpoint}")
        else:
            print("ERROR: No s'ha especificat cap --checkpoint i no se n'ha trobat cap a 'checkpoints/'")
            sys.exit(1)

    return args


def load_model(ckpt_path: str, vocab: Vocabulary, device: torch.device):
    """Carrega el model des del checkpoint guardat durant el train."""
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]

    encoder = EncoderCNN(a["embed_size"], backbone=a["backbone"]).to(device)
    decoder = DecoderRNN(a["embed_size"], a["hidden_size"], len(vocab),
                         a["num_layers"], max_seq_length=25).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder


@torch.no_grad()
def generate_caption(image_path: str, encoder, decoder, vocab, transform, device) -> str:
    """Genera un peu de foto per a una sola imatge."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    features = encoder(x)
    ids = decoder.sample(features).cpu().numpy()[0].tolist()
    return vocab.decode(ids)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usant dispositiu: {device}")

    # ---- Carreguem vocabulari i model ----
    print("Carregant vocabulari i model...")
    with open(args.vocab, "rb") as f:
        vocab = pickle.load(f)
    encoder, decoder = load_model(args.checkpoint, vocab, device)
    transform = get_transform(train=False)

    # ---- Carreguem les imatges del conjunt de TEST ----
    print("Carregant ids del conjunt de test...")
    _, _, test_ids = split_image_ids(args.captions_csv)

    if args.num_images is not None:
        test_ids = test_ids[:args.num_images]
        print(f"  (Avaluant només {args.num_images} imatges per anar ràpid)")

    # ---- Carreguem tots els peus de foto reals (per comparar) ----
    # Cada imatge té 5 peus de foto originals -> guadem-los tots
    df = pd.read_csv(args.captions_csv)
    all_captions = defaultdict(list)
    for _, row in df.iterrows():
        tokens = simple_tokenize(str(row["caption"]))
        all_captions[row["image"]].append(tokens)

    # ---- Generem i comparem ----
    print(f"\nGenerant peus de foto per a {len(test_ids)} imatges de test...")
    references = []   # llista de llistes: per cada imatge, tots els seus captions originals
    hypotheses = []   # caption generat per la xarxa per cada imatge

    images_dir = Path(args.images_dir)
    smoother = SmoothingFunction().method4  # suavitzat per evitar problemes amb BLEU-4

    for img_name in tqdm(test_ids):
        img_path = images_dir / img_name
        if not img_path.exists():
            continue

        # Generar caption amb la xarxa
        generated = generate_caption(str(img_path), encoder, decoder, vocab, transform, device)
        hyp_tokens = simple_tokenize(generated)

        # Captions originals de referència per a aquesta imatge
        refs = all_captions.get(img_name, [])
        if not refs:
            continue

        references.append(refs)
        hypotheses.append(hyp_tokens)

    # ---- Calculem els BLEU scores ----
    print(f"\n{'='*50}")
    print("RESULTATS - BLEU SCORE")
    print(f"{'='*50}")
    print(f"Imatges avaluades: {len(hypotheses)}")

    # corpus_bleu calcula la mètrica sobre tot el corpus (més robusta que per imatge)
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    print(f"BLEU-1: {bleu1:.4f}  ({bleu1*100:.1f}%)")
    print(f"BLEU-2: {bleu2:.4f}  ({bleu2*100:.1f}%)")
    print(f"BLEU-4: {bleu4:.4f}  ({bleu4*100:.1f}%)  <-- el més important")
    print(f"{'='*50}")

    # Referència: papers de la literatura solen tenir BLEU-4 entre 0.20 i 0.35 per Flickr8k
    if bleu4 >= 0.25:
        print("  → El model ha après força bé! BLEU-4 >= 0.25 és un bon resultat.")
    elif bleu4 >= 0.15:
        print("  → Resultat acceptable. Més èpoques o un backbone millor podrien ajudar.")
    else:
        print("  → El model encara aprèn poc. Prova amb més èpoques d'entrenament.")

    # ---- Mostrem alguns exemples per veure'ls ----
    print(f"\nEXEMPLES (primeres 5 imatges):")
    print("-" * 50)
    for img_name, hyp, refs in zip(test_ids[:5], hypotheses[:5], references[:5]):
        ref_example = " ".join(refs[0])
        print(f"Imatges:    {img_name}")
        print(f"Original:  {ref_example}")
        print(f"Generat:   {' '.join(hyp)}")
        print()


if __name__ == "__main__":
    main()
