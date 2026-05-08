"""
sample.py  —  Baseline (src/baseline/sample.py)
================================================
Genera la caption d'una imatge usant el model base (CNN + LSTM sense atenció).

Funcions principals:
    load_checkpoint:    reconstrueix el model i carrega els pesos entrenats
    caption_image:      genera una caption donada la ruta d'una imatge
    caption_pil_image:  igual però accepta un PIL.Image directament
                        (útil per al dataset HuggingFace, que retorna PIL directament)
    main:               punt d'entrada quan s'executa des del terminal

Diferència respecte al sample.py del model amb atenció:
    Baseline:  usa DecoderRNN.sample() → decodificació greedy (una sola seqüència)
    Atenció:   usa AttentionDecoder.beam_search() → manté múltiples hipòtesis
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image

from src.shared.dataset    import get_transform
from src.baseline.model    import DecoderRNN, EncoderCNN
from src.shared.vocabulary import Vocabulary


# ─────────────────────────────────────────────────────────────────────────────
# CÀRREGA DEL MODEL ENTRENAT
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, vocab_path: str, device: torch.device):
    """
    Reconstrueix l'encoder i el decoder a partir d'un checkpoint entrenat.

    Un checkpoint (.pt) és un fitxer que conté:
        - "encoder":    pesos de l'EncoderCNN (state_dict)
        - "decoder":    pesos del DecoderRNN (state_dict)
        - "args":       els hiperparàmetres usats durant l'entrenament
        - "vocab_size": mida del vocabulari
        - "epoch":      en quina època es va guardar

    Per poder reconstruir el model EXACTAMENT igual que durant l'entrenament,
    hem de llegir els hiperparàmetres del checkpoint (no els de la comanda actual).

    Args:
        ckpt_path:  ruta al fitxer .pt (ex: "checkpoints/ckpt_best.pt")
        vocab_path: ruta al fitxer .pkl del vocabulari
        device:     on carregar el model (cpu o cuda)

    Returns:
        encoder, decoder, vocab — tots en mode eval()
    """
    # ── Carrega el vocabulari ─────────────────────────────────────────────────
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    # pickle.load deserialitza l'objecte Vocabulary complet des del fitxer binari.

    # ── Carrega el checkpoint ─────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    # map_location=device: si el checkpoint es va guardar en GPU però ara tenim CPU
    # (o viceversa), PyTorch el carrega al device correcte automàticament.

    a = ckpt["args"]
    # a és el diccionari d'arguments amb tots els hiperparàmetres de l'entrenament:
    # a["embed_size"], a["hidden_size"], a["backbone"], a["num_layers"], etc.
    # És CRUCIAL usar els mateixos hiperparàmetres que durant l'entrenament,
    # o els pesos carregats no s'ajustarien a l'arquitectura.

    # ── Reconstrueix l'encoder ────────────────────────────────────────────────
    encoder = EncoderCNN(
        embed_size=a["embed_size"],
        backbone=a["backbone"],
    ).to(device).eval()
    # .to(device): mou el model al dispositiu correcte (CPU o GPU)
    # .eval(): posa el model en mode avaluació:
    #          → dropout s'inactiva (no s'apaguen neurones)
    #          → batch normalization usa les estadístiques apreses (no les del mini-batch)

    # ── Reconstrueix el decoder ────────────────────────────────────────────────
    decoder = DecoderRNN(
        embed_size=a["embed_size"],
        hidden_size=a["hidden_size"],
        vocab_size=len(vocab),
        num_layers=a["num_layers"],
    ).to(device).eval()

    # ── Carrega els pesos entrenats ────────────────────────────────────────────
    encoder.load_state_dict(ckpt["encoder"])
    # load_state_dict: copia els tensors del checkpoint a les capes del model.
    # ckpt["encoder"] és un OrderedDict {nom_capa: tensor_de_pesos}.

    decoder.load_state_dict(ckpt["decoder"])

    return encoder, decoder, vocab


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓ DE CAPTIONS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def caption_image(image_path: str, encoder, decoder, vocab, device) -> str:
    """
    Genera la caption d'una imatge donada la seva ruta al disc.

    @torch.no_grad(): desactiva el càlcul de gradients durant tota la funció.
    Estalvia memòria i accelera l'execució (no necessitem gradients a la inferència).

    Args:
        image_path: ruta completa a la imatge .jpg (ex: "dataset/Images/abc123.jpg")
        encoder:    EncoderCNN en mode eval()
        decoder:    DecoderRNN en mode eval()
        vocab:      vocabulari per convertir índexs a text
        device:     cpu o cuda

    Returns:
        caption com a string (ex: "a dog running on the grass")
    """
    tfm = get_transform(train=False)
    # Obté les transformacions de VALIDACIÓ (sense augmentation):
    # Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)
    # Hem d'usar exactament les mateixes transformacions que durant l'entrenament,
    # o les imatges tindran una distribució diferent a la que el model va aprendre.

    img = Image.open(image_path).convert("RGB")
    # PIL.Image.open: obre la imatge des del disc (qualsevol format: jpg, png, etc.)
    # .convert("RGB"): assegura que la imatge té 3 canals.
    #   - Imatges en escala de grisos (1 canal) → convertides a 3 canals idèntics
    #   - Imatges RGBA (4 canals, amb transparència) → es descarta el canal alfa

    x = tfm(img).unsqueeze(0).to(device)
    # tfm(img):    PIL Image → tensor [3, 224, 224]  (float, normalitzat)
    # unsqueeze(0): [3, 224, 224] → [1, 3, 224, 224]  (afegeix dimensió de batch)
    #              L'encoder espera [B, 3, 224, 224], i aquí B=1.
    # .to(device): mou el tensor a la GPU si está disponible

    feat = encoder(x)
    # Passa la imatge per l'EncoderCNN.
    # x [1, 3, 224, 224] → feat [1, embed_size]  (vector global de la imatge)

    ids = decoder.sample(feat).cpu().numpy()[0].tolist()
    # decoder.sample(feat):  greedy decoding → tensor [1, max_seq_length]
    # .cpu():                mou el tensor a CPU (necessari si estem a GPU, per a numpy)
    # .numpy():              tensor → array numpy [1, max_seq_length]
    # [0]:                   selecciona la primera (i única) fila → array [max_seq_length]
    # .tolist():             array numpy → llista Python d'enters

    return vocab.decode(ids)
    # Converteix la llista d'índexs a text:
    # [1, 8, 4, 23, 2, 0, 0, ...] → "a dog runs"
    # (omet <pad>, s'atura a <end>, omet <start>)


@torch.no_grad()
def caption_pil_image(pil_img, encoder, decoder, vocab, device) -> str:
    """Genera la caption d'una imatge a partir d'un objecte PIL.Image.

    Equivalent a caption_image() però accepta directament un PIL.Image
    en lloc d'una ruta de fitxer.

    Per a qué serveix?
        El dataset Flickr30k de HuggingFace retorna les imatges ja carregades
        com a PIL.Image, no com a rutes. Crear fitxers temporals per a cada
        imatge seria molt lent. Aquesta funció evita el pas intermedi.

    Args:
        pil_img: objecte PIL.Image (la imatge ja carregada en memòria)
        encoder, decoder, vocab, device: igual que caption_image()

    Returns:
        caption com a string
    """
    tfm  = get_transform(train=False)
    x    = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    # Mateix procés que caption_image(), però sense PIL.Image.open()
    # perquè ja tenim la imatge en memòria.

    feat = encoder(x)
    ids  = decoder.sample(feat).cpu().numpy()[0].tolist()
    return vocab.decode(ids)


# ─────────────────────────────────────────────────────────────────────────────
# PUNT D'ENTRADA TERMINAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Genera la caption d'una sola imatge des del terminal.

    Ús:
        python -m src.baseline.sample \\
            --image dataset/Images/dog.jpg \\
            --checkpoint checkpoints_baseline/ckpt_best.pt \\
            --vocab dataset/vocab.pkl
    """
    p = argparse.ArgumentParser(description="Genera una caption per a una imatge (baseline).")
    p.add_argument("--image",      required=True,
                   help="Ruta a la imatge .jpg")
    p.add_argument("--checkpoint", required=True,
                   help="Ruta al checkpoint .pt del model entrenat")
    p.add_argument("--vocab",      default="data/flickr8k/vocab.pkl",
                   help="Ruta al fitxer .pkl del vocabulari")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Usa GPU si está disponible (molt més ràpid per a la inferència)

    encoder, decoder, vocab = load_checkpoint(args.checkpoint, args.vocab, device)
    cap = caption_image(args.image, encoder, decoder, vocab, device)
    print(f"{Path(args.image).name}: {cap}")
    # Exemple de sortida: "dog.jpg: a brown dog is running through a field"


if __name__ == "__main__":
    main()