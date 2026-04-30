"""BLEU evaluation on the full test set using the attention model, logged to wandb."""
import pandas as pd
import torch
import wandb
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from src.shared.dataset import split_image_ids
from src.attention.sample import caption_image, load_checkpoint
from src.shared.vocabulary import simple_tokenize

CHECKPOINT   = "checkpoints_attention/ckpt_best.pt"
VOCAB_PATH   = "data/flickr8k/vocab.pkl"
IMAGES_DIR   = "data/flickr8k/Images"
CAPTIONS_CSV = "data/flickr8k/captions.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, decoder, vocab = load_checkpoint(CHECKPOINT, VOCAB_PATH, device)

_, _, test_ids = split_image_ids(CAPTIONS_CSV)

df = pd.read_csv(CAPTIONS_CSV)
smooth = SmoothingFunction().method1

run = wandb.init(entity="learning6", project="image-captioning",
                 name="bleu-eval-fulltest-attention", config={"n_images": len(test_ids), "checkpoint": CHECKPOINT})

all_refs, all_hyps = [], []
table = wandb.Table(columns=["image", "generated_caption", "reference_captions", "BLEU-1", "BLEU-4"])

print(f"Evaluating {len(test_ids)} test images (attention model)...")
print(f"{'Image':<35} {'BLEU-1':>7} {'BLEU-4':>7}  Caption")
print("-" * 100)

for img in test_ids:
    refs = [simple_tokenize(c) for c in df[df["image"] == img]["caption"].tolist()]
    hyp  = simple_tokenize(caption_image(f"{IMAGES_DIR}/{img}", encoder, decoder, vocab, device))

    b1 = sentence_bleu(refs, hyp, weights=(1,0,0,0), smoothing_function=smooth)
    b4 = sentence_bleu(refs, hyp, weights=(.25,.25,.25,.25), smoothing_function=smooth)

    all_refs.append(refs)
    all_hyps.append(hyp)

    ref_str = " | ".join([" ".join(r) for r in refs])
    table.add_data(wandb.Image(f"{IMAGES_DIR}/{img}"), " ".join(hyp), ref_str, round(b1, 3), round(b4, 3))
    print(f"{img:<35} {b1:>7.3f} {b4:>7.3f}  {' '.join(hyp)}")

cb1 = corpus_bleu(all_refs, all_hyps, weights=(1,0,0,0))
cb4 = corpus_bleu(all_refs, all_hyps, weights=(.25,.25,.25,.25))
print("-" * 100)
print(f"{'Corpus BLEU':<35} {cb1:>7.3f} {cb4:>7.3f}")

wandb.log({"bleu/corpus_bleu1": cb1, "bleu/corpus_bleu4": cb4, "eval_table": table})
print(f"\nWandb: {run.url}")
wandb.finish()
