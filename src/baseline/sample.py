"""Generate a caption for a single image using a trained checkpoint."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image

from dataset import get_transform # per fer les transformacions de la imatge
from model import DecoderRNN, EncoderCNN 
from vocabulary import Vocabulary  # noqa: F401  (needed for pickle load)


def load_checkpoint(ckpt_path: str, vocab_path: str, device: torch.device):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f) # carrega vocabulari
    ckpt = torch.load(ckpt_path, map_location=device) # recupera els checkpoints
    a = ckpt["args"] # del chechpoint agafa els hiperperàmetres d'aquell entrenament
    encoder = EncoderCNN(a["embed_size"], backbone=a["backbone"]).to(device).eval() # reconstrueix l'encoder i es posa en mode eval
    decoder = DecoderRNN(a["embed_size"], a["hidden_size"], len(vocab), # i el decoder
                         a["num_layers"]).to(device).eval() # i mode eval
    encoder.load_state_dict(ckpt["encoder"]) # carrega pesos entrenats a l'encoder
    decoder.load_state_dict(ckpt["decoder"]) # carrega pesos
    return encoder, decoder, vocab


@torch.no_grad() # no volem fer backprop. Aquesta seguent funció genera la caption d'una image.
def caption_image(image_path: str, encoder, decoder, vocab, device) -> str:
    tfm = get_transform(train=False) # crea les transformacions
    img = Image.open(image_path).convert("RGB") # obre la imatge i la converteix a 3 canals (input esperat)
    x = tfm(img).unsqueeze(0).to(device) # aplica les transformacions a la imatge
    feat = encoder(x) # troba el vector de característiques de la imatge passant-la per l'encoder
    ids = decoder.sample(feat).cpu().numpy()[0].tolist() # crea la caption (lista d'ids) del vector de característiques passant-lo pel decoder
    return vocab.decode(ids) # tradueix la llista d'ids a text


def main():
    p = argparse.ArgumentParser() # arguments
    p.add_argument("--image", required=True) # paht de la imarge que vols captionar
    p.add_argument("--checkpoint", required=True) # quin model entrenat vols carregar
    p.add_argument("--vocab", default="data/flickr8k/vocab.pkl") # path del vocab
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # assigna device gpu i si no pots cpu
    encoder, decoder, vocab = load_checkpoint(args.checkpoint, args.vocab, device) # funció load_chechpoint per crear el model
    cap = caption_image(args.image, encoder, decoder, vocab, device) # funció que et retorna la caption de la imarge
    print(f"{Path(args.image).name}: {cap}") # mostra el nom de la imatge i la caption generada.


if __name__ == "__main__":
    main()
