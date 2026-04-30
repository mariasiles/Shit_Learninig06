"""CNN encoder with spatial features + LSTM decoder with Bahdanau attention and beam search.

Based on "Show, Attend and Tell" (Xu et al., 2015).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNNAttention(nn.Module):
    """ResNet-50 that outputs a spatial grid of feature vectors instead of a single embedding."""

    def __init__(self, backbone: str = "resnet50"):
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            net = models.resnet50(weights=weights)
        elif backbone == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V2
            net = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Drop avgpool and fc — keep spatial feature map [B, 2048, 7, 7]
        self.cnn = nn.Sequential(*list(net.children())[:-2])
        self.encoder_dim = 2048

        for p in self.cnn.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.cnn(images)            # [B, 2048, 7, 7]
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1)         # [B, 7, 7, 2048]
        return out.view(B, H * W, C)          # [B, 49, 2048]


class Attention(nn.Module):
    """Bahdanau (additive) attention over encoder spatial features."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.enc_att = nn.Linear(encoder_dim, attention_dim)
        self.dec_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, h: torch.Tensor):
        # encoder_out: [B, num_pixels, encoder_dim]
        # h:           [B, decoder_dim]
        e = self.full_att(torch.tanh(
            self.enc_att(encoder_out) + self.dec_att(h).unsqueeze(1)
        )).squeeze(2)                          # [B, num_pixels]
        alpha = self.softmax(e)               # [B, num_pixels]
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [B, encoder_dim]
        return context, alpha


class AttentionDecoder(nn.Module):
    """LSTM decoder with attention. Supports teacher-forcing training and beam search."""

    def __init__(
        self,
        encoder_dim: int = 2048,
        embed_size: int = 256,
        hidden_size: int = 512,
        vocab_size: int = 10000,
        attention_dim: int = 256,
        dropout: float = 0.5,
        max_seq_length: int = 20,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def _init_hidden(self, encoder_out: torch.Tensor):
        mean = encoder_out.mean(dim=1)         # [B, encoder_dim]
        return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor, lengths: list[int]):
        B = encoder_out.size(0)
        embeddings = self.dropout(self.embed(captions))   # [B, T, embed_size]
        h, c = self._init_hidden(encoder_out)

        decode_lengths = [l - 1 for l in lengths]
        max_t = max(decode_lengths)

        preds = []
        for t in range(max_t):
            bt = sum(1 for l in decode_lengths if l > t)
            context, _ = self.attention(encoder_out[:bt], h[:bt])
            lstm_in = torch.cat([embeddings[:bt, t], context], dim=1)
            h_new, c_new = self.lstm_cell(lstm_in, (h[:bt], c[:bt]))
            preds.append(self.fc(self.dropout(h_new)))    # [bt, vocab_size]
            if bt < B:
                h = torch.cat([h_new, h[bt:]], dim=0)
                c = torch.cat([c_new, c[bt:]], dim=0)
            else:
                h, c = h_new, c_new

        return torch.cat(preds, dim=0)

    @torch.no_grad()
    def beam_search(
        self,
        encoder_out: torch.Tensor,
        start_idx: int,
        end_idx: int,
        beam_size: int = 3,
    ) -> list[int]:
        """Beam search for a single image. Returns token ids (no special tokens)."""
        device = encoder_out.device
        k = beam_size

        enc = encoder_out.expand(k, -1, -1)               # [k, num_pixels, enc_dim]
        h, c = self._init_hidden(enc)                      # [k, hidden_size]

        seqs = torch.full((k, 1), start_idx, dtype=torch.long, device=device)
        scores = torch.zeros(k, device=device)

        complete_seqs, complete_scores = [], []

        for step in range(self.max_seq_length):
            embeddings = self.embed(seqs[:, -1])            # [k, embed_size]
            context, _ = self.attention(enc, h)            # [k, enc_dim]
            lstm_in = torch.cat([embeddings, context], dim=1)
            h, c = self.lstm_cell(lstm_in, (h, c))

            log_probs = torch.log_softmax(self.fc(h), dim=1)  # [k, vocab_size]
            total = scores.unsqueeze(1) + log_probs            # [k, vocab_size]

            if step == 0:
                top_scores, top_words = total[0].topk(k)
            else:
                top_scores, top_words = total.view(-1).topk(k)

            beam_idx = top_words // self.vocab_size
            word_idx = top_words % self.vocab_size

            seqs = torch.cat([seqs[beam_idx], word_idx.unsqueeze(1)], dim=1)
            h, c = h[beam_idx], c[beam_idx]
            enc = enc[beam_idx]
            scores = top_scores

            still_running = []
            for j in range(k):
                if word_idx[j].item() == end_idx:
                    complete_seqs.append(seqs[j, 1:-1].tolist())  # strip <start>/<end>
                    complete_scores.append(scores[j].item())
                else:
                    still_running.append(j)

            if not still_running:
                break

            k = len(still_running)
            seqs = seqs[still_running]
            h, c = h[still_running], c[still_running]
            enc = enc[still_running]
            scores = scores[still_running]

        if not complete_seqs:
            complete_seqs = [seqs[0, 1:].tolist()]
            complete_scores = [scores[0].item()]

        best = max(range(len(complete_scores)), key=lambda i: complete_scores[i])
        return complete_seqs[best]
