"""
sample.py  —  Model amb Atenció (src/attention/sample.py)
==========================================================
Genera la caption d'una imatge usant el model amb atenció i beam search.

Diferències clau respecte al sample.py del baseline:
    Encoder:  EncoderCNNAttention en lloc de EncoderCNN
              → retorna [1, 49, 2048] en lloc de [1, embed_size]
    Decoder:  AttentionDecoder.beam_search() en lloc de DecoderRNN.sample()
              → manté beam_size hipòtesis en paral·lel en lloc d'agafar sempre el màxim

Funcions:
    load_checkpoint:    reconstrueix el model amb atenció i carrega els pesos
    caption_image:      genera caption des d'una ruta de fitxer (amb beam search)
    caption_pil_image:  igual però accepta PIL.Image (per a HuggingFace)
    main:               punt d'entrada terminal
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image

from src.shared.dataset    import get_transform
from src.attention.model   import AttentionDecoder, EncoderCNNAttention
from src.shared.vocabulary import Vocabulary  # noqa: F401 — necessari per a pickle.load()


# ─────────────────────────────────────────────────────────────────────────────
# CÀRREGA DEL MODEL ENTRENAT
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, vocab_path: str, device: torch.device):
    """Reconstrueix l'encoder i el decoder amb atenció des d'un checkpoint.

    Diferència amb load_checkpoint del baseline:
        - Usa EncoderCNNAttention (en lloc de EncoderCNN)
        - Usa AttentionDecoder (en lloc de DecoderRNN)
        - Necessita attention_dim com a hiperparàmetre addicional

    Args:
        ckpt_path:  ruta al checkpoint .pt
        vocab_path: ruta al vocabulari .pkl
        device:     cpu o cuda

    Returns:
        encoder, decoder, vocab — en mode eval()
    """
    # ── Carrega el vocabulari ─────────────────────────────────────────────────
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    # Deserialitza l'objecte Vocabulary complet (amb els dos diccionaris word2idx i idx2word).

    # ── Carrega el checkpoint ─────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    # map_location: carrega al device correcte independentment d'on es va guardar.
    # Si el checkpoint es va entrenar a GPU i ara executem a CPU, funciona igualment.

    a = ckpt["args"]
    # Diccionari amb TOTS els hiperparàmetres de l'entrenament original.
    # Exemple: {"embed_size": 300, "hidden_size": 512, "attention_dim": 256,
    #           "dropout": 0.5, "backbone": "resnet152", ...}

    # ── Reconstrueix l'encoder amb atenció ────────────────────────────────────
    encoder = EncoderCNNAttention(backbone=a["backbone"]).to(device).eval()
    # Nota: EncoderCNNAttention NO rep embed_size com a paràmetre.
    # A diferència del baseline (que afegeix una capa lineal 2048→embed_size),
    # l'encoder amb atenció sempre retorna [B, 49, 2048] independentment de l'embed_size.
    # La projecció de dimensió la fa el mòdul d'atenció intern.

    # ── Reconstrueix el decoder amb atenció ───────────────────────────────────
    decoder = AttentionDecoder(
        encoder_dim=2048,                    # fixat per l'arquitectura ResNet
        embed_size=a["embed_size"],          # 256 o la dim de GloVe/Word2Vec
        hidden_size=a["hidden_size"],        # ex: 512
        vocab_size=len(vocab),               # ~3000 per a Flickr8k
        attention_dim=a["attention_dim"],    # ex: 256 (dimensió interna de l'atenció)
        dropout=a["dropout"],               # ex: 0.5 (inactiu en mode eval)
    ).to(device).eval()

    # ── Carrega els pesos entrenats ────────────────────────────────────────────
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    # load_state_dict: copia cada tensor de pesos del checkpoint a la capa corresponent.
    # Si hi ha cap discrepància de mida o nom de capa, Python llança un error.

    return encoder, decoder, vocab


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓ DE CAPTIONS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def caption_image(
    image_path: str,
    encoder,
    decoder,
    vocab,
    device,
    beam_size: int = 3,
) -> str:
    """Genera la caption d'una imatge usant beam search.

    @torch.no_grad(): desactiva el còmput de gradients en tota la funció.
    No els necessitem a la inferència; sense ells s'estalvia memòria i temps.

    Diferència clau vs baseline (greedy):
        Greedy:      en cada pas, escull el token amb probabilitat màxima.
                     → Ràpid però pot perdre seqüències globalment millors.
        Beam search: manté beam_size camins actius; en cada pas, expandeix TOTS
                     i queda amb els millors (major log-probabilitat acumulada).
                     → Millors resultats però (beam_size) vegades més costós.

    Args:
        image_path: ruta a la imatge .jpg
        encoder:    EncoderCNNAttention en mode eval()
        decoder:    AttentionDecoder en mode eval()
        vocab:      vocabulari
        device:     cpu o cuda
        beam_size:  nombre de camins paral·lels al beam search (defecte 3)

    Returns:
        caption com a string (ex: "two dogs playing in the park")
    """
    tfm = get_transform(train=False)
    # Transformacions de validació: Resize(256) → CenterCrop(224) → ToTensor → Normalize
    # IMPORTANT: han de ser les mateixes que durant l'entrenament.

    img = Image.open(image_path).convert("RGB")
    # Obre la imatge i assegura 3 canals RGB.

    x = tfm(img).unsqueeze(0).to(device)
    # PIL Image → tensor [3, 224, 224] → [1, 3, 224, 224] → GPU/CPU

    features = encoder(x)
    # Passa la imatge per l'encoder amb atenció.
    # x [1, 3, 224, 224] → features [1, 49, 2048]
    # 49 = 7×7 regions de la imatge, cadascuna amb 2048 dimensions.
    # A diferència del baseline que retorna [1, embed_size],
    # aquí tenim la graella espacial completa per al mecanisme d'atenció.

    ids = decoder.beam_search(
        features,
        start_idx=vocab.word2idx["<start>"],  # índex del token d'inici (1)
        end_idx=vocab.word2idx["<end>"],      # índex del token de fi (2)
        beam_size=beam_size,                  # nombre de camins paral·lels
    )
    # decoder.beam_search retorna una llista d'enters (sense <start> ni <end>)
    # Exemple: [8, 4, 23, 87]  → es decodificarà com "a dog runs fast"

    return vocab.decode(ids, skip_special=False)
    # skip_special=False perquè beam_search ja ha eliminat <start> i <end>.
    # vocab.decode converteix [8, 4, 23, 87] → "a dog runs fast"


@torch.no_grad()
def caption_pil_image(
    pil_img,
    encoder,
    decoder,
    vocab,
    device,
    beam_size: int = 3,
) -> str:
    """Genera la caption d'un PIL.Image directament (sense llegir del disc).

    Mateixa lògica que caption_image() però rep la imatge ja carregada
    en memòria (objecte PIL.Image) en lloc d'una ruta de fitxer.

    Per a qué serveix?
        HuggingFace retorna les imatges com a PIL.Image, no com a rutes.
        La funció test_pil[img] al train.py retorna directament un PIL.Image.
        Usar caption_image() obligaria a guardar la imatge temporalment al disc
        i tornar-la a obrir, cosa que seria molt ineficient.

    Args:
        pil_img:   objecte PIL.Image (imatge ja en memòria)
        beam_size: nombre de camins per al beam search

    Returns:
        caption com a string
    """
    tfm      = get_transform(train=False)
    x        = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    # .convert("RGB") és necessari perquè el PIL.Image de HuggingFace pot
    # venir en mode L (escala de grisos), RGBA o altres formats.

    features = encoder(x)
    # [1, 3, 224, 224] → [1, 49, 2048]

    ids = decoder.beam_search(
        features,
        start_idx=vocab.word2idx["<start>"],
        end_idx=vocab.word2idx["<end>"],
        beam_size=beam_size,
    )
    return vocab.decode(ids, skip_special=False)


# ─────────────────────────────────────────────────────────────────────────────
# PUNT D'ENTRADA TERMINAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Genera la caption d'una sola imatge des del terminal amb beam search.

    Ús:
        python -m src.attention.sample \\
            --image dataset/Images/dog.jpg \\
            --checkpoint checkpoints_attention/ckpt_best.pt \\
            --vocab dataset/vocab.pkl \\
            --beam-size 3
    """
    p = argparse.ArgumentParser(description="Genera una caption (attention + beam search).")
    p.add_argument("--image",      required=True,
                   help="Ruta a la imatge .jpg")
    p.add_argument("--checkpoint", required=True,
                   help="Ruta al checkpoint .pt del model entrenat")
    p.add_argument("--vocab",      default="data/flickr8k/vocab.pkl",
                   help="Ruta al vocabulari .pkl")
    p.add_argument("--beam-size",  type=int, default=3,
                   help="Nombre de camins del beam search (3-5 habitual, >5 molt lent)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder, vocab = load_checkpoint(args.checkpoint, args.vocab, device)

    cap = caption_image(
        args.image, encoder, decoder, vocab, device,
        beam_size=args.beam_size,
    )
    print(f"{Path(args.image).name}: {cap}")
    # Exemple de sortida: "dog.jpg: a dog is running through a green field"


if __name__ == "__main__":
    main()