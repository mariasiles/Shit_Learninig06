"""CNN encoder with spatial features + LSTM decoder with Bahdanau attention and beam search.

Based on "Show, Attend and Tell" (Xu et al., 2015).
"""
# Paper original on es basa aquest model: Xu et al. 2015

from __future__ import annotations  # per escriure anotacions modernes

import torch
import torch.nn as nn
import torchvision.models as models


# ════════════════════════════════════════════════════════
# 1. ENCODER
# ════════════════════════════════════════════════════════

class EncoderCNNAttention(nn.Module):
    """ResNet-50 that outputs a spatial grid of feature vectors instead of a single embedding."""

    def __init__(self, backbone: str = "resnet50"):
        super().__init__()
        if backbone == "resnet50":
            try:
                net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            except AttributeError:
                net = models.resnet50(pretrained=True)
        elif backbone == "resnet152":
            try:
                net = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
            except AttributeError:
                net = models.resnet152(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Drop avgpool and fc — keep spatial feature map [B, 2048, 7, 7]
        self.cnn = nn.Sequential(*list(net.children())[:-2])
        # list(net.children()) → llista de totes les capes de la ResNet
        # [:-2] → elimina les 2 últimes capes (avgpool i fc)
        # el baseline eliminava [:-1] (només la fc) → obtenia [B, 2048, 1, 1]
        # aquí eliminem també avgpool → obtenim [B, 2048, 7, 7] (mapa espacial!)
        self.encoder_dim = 2048  # dimensions de cada vector de regió

        for p in self.cnn.parameters():
            p.requires_grad = False  # congela la CNN, no entrena els seus pesos
        self.finetuning = False  # quan True, layer4 rep gradients

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.finetuning:
            with torch.no_grad():
                out = self.cnn[:-1](images)  # totes les capes menys layer4 [B, 1024, 14, 14]
            out = self.cnn[-1](out)          # layer4 amb gradients [B, 2048, 7, 7]
        else:
            with torch.no_grad():
                out = self.cnn(images)       # CNN completament congelada [B, 2048, 7, 7]
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1)
        return out.view(B, H * W, C)         # [B, 49, 2048]
        # cada imatge queda representada per 49 vectors de 2048 dimensions
        # cada vector = una regió de la imatge


# ════════════════════════════════════════════════════════
# 2. MÒDUL D'ATENCIÓ (Bahdanau)
# ════════════════════════════════════════════════════════

class Attention(nn.Module):
    """Bahdanau (additive) attention over encoder spatial features."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.enc_att = nn.Linear(encoder_dim, attention_dim)
        # projecta cada una de les 49 regions de [2048] a [attention_dim=256]
        self.dec_att = nn.Linear(decoder_dim, attention_dim)
        # projecta l'estat ocult h del decoder de [512] a [256]
        self.full_att = nn.Linear(attention_dim, 1)
        # projecta de [256] a [1] → un score per cada regió
        self.softmax = nn.Softmax(dim=1)
        # normalitza els scores de les 49 regions perquè sumin 1

    def forward(self, encoder_out: torch.Tensor, h: torch.Tensor):
        # encoder_out: [B, 49, 2048] → les 49 regions de la imatge
        # h:           [B, 512]      → l'estat ocult actual del decoder
        e = self.full_att(torch.tanh(
            self.enc_att(encoder_out) + self.dec_att(h).unsqueeze(1)
            # self.enc_att(encoder_out) → [B, 49, 256] info de cada regió
            # self.dec_att(h).unsqueeze(1) → [B, 1, 256] info del decoder (s'expandeix a totes les regions)
            # tanh combina les dues informacions → [B, 49, 256]
            # full_att redueix a → [B, 49, 1]
        )).squeeze(2)                          # [B, num_pixels] → elimina última dim
        # cada valor és "quant important és aquesta regió ara"
        alpha = self.softmax(e)               # [B, num_pixels] → pesos d'atenció, sumen 1
        # ex: regió del gos té alpha=0.8, fons té alpha=0.01...
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [B, encoder_dim]
        # alpha.unsqueeze(2) → [B, 49, 1] per poder multiplicar
        # encoder_out * alpha → [B, 49, 2048] cada regió ponderada pel seu pes
        # .sum(dim=1) → [B, 2048] suma les 49 regions → vector de context
        return context, alpha
        # context: resum ponderat de la imatge per generar la paraula actual
        # alpha: els pesos (útils per visualitzar on "mira" el model)


# ════════════════════════════════════════════════════════
# 3. DECODER AMB ATENCIÓ
# ════════════════════════════════════════════════════════

class AttentionDecoder(nn.Module):
    """LSTM decoder with attention. Supports teacher-forcing training and beam search."""

    def __init__(
        self,
        encoder_dim: int = 2048,   # mida dels vectors de les regions
        embed_size: int = 256,     # mida dels embeddings de les paraules
        hidden_size: int = 512,    # mida de l'estat ocult de la LSTM
        vocab_size: int = 10000,   # nombre de paraules del vocabulari
        attention_dim: int = 256,  # dimensió interna del mòdul d'atenció
        dropout: float = 0.5,      # probabilitat de dropout
        max_seq_length: int = 20,  # longitud màxima de la caption generada
        pretrained_weights: "torch.Tensor | None" = None,  # pesos GloVe [vocab_size, embed_size]
        freeze_embeddings: bool = False,  # si True, els pesos GloVe no s'actualitzen
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        # mòdul d'atenció que calcula el context a cada pas
        self.embed = nn.Embedding(vocab_size, embed_size)
        # taula d'embedding: índex → vector de 256 dims
        if pretrained_weights is not None:
            self.embed.weight = nn.Parameter(pretrained_weights)
        if freeze_embeddings:
            self.embed.weight.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        # regularització: apaga neurones aleatòriament durant entrenament
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        # LSTMCell (no LSTM!) processa UN sol pas a la vegada
        # input_size = embed_size(256) + encoder_dim(2048) = 2304
        # → perquè cada pas rep: embedding de la paraula + context de l'atenció
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        # dues capes per inicialitzar h i c de la LSTM a partir de la imatge
        # (en el baseline la imatge s'injectava com a primer token, aquí s'usa per inicialitzar)
        self.fc = nn.Linear(hidden_size, vocab_size)
        # capa final: hidden [512] → probabilitats sobre vocabulari [2982]

    def _init_hidden(self, encoder_out: torch.Tensor):
        mean = encoder_out.mean(dim=1)         # [B, encoder_dim] → mitjana de les 49 regions
        return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))
        # retorna h i c inicials [B, 512] calculats a partir de la imatge

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor, lengths: list[int]):
        # encoder_out: [B, 49, 2048]
        # captions:    [B, T] tokens (inclou <start>)
        # lengths:     longitud real de cada caption
        B = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        embeddings = self.dropout(self.embed(captions))   # [B, T, embed_size]
        h, c = self._init_hidden(encoder_out)

        decode_lengths = [l - 1 for l in lengths]
        max_t = max(decode_lengths)

        # Acumula els pesos d'atenció per a la Doubly Stochastic regularització
        alphas_sum = torch.zeros(B, num_pixels, device=encoder_out.device)

        preds = []
        for t in range(max_t):
            bt = sum(1 for l in decode_lengths if l > t)
            context, alpha = self.attention(encoder_out[:bt], h[:bt])  # alpha: [bt, num_pixels]
            alphas_sum[:bt] = alphas_sum[:bt] + alpha  # suma sobre el temps
            lstm_in = torch.cat([embeddings[:bt, t], context], dim=1)
            h_new, c_new = self.lstm_cell(lstm_in, (h[:bt], c[:bt]))
            preds.append(self.fc(self.dropout(h_new)))
            if bt < B:
                h = torch.cat([h_new, h[bt:]], dim=0)
                c = torch.cat([c_new, c[bt:]], dim=0)
            else:
                h, c = h_new, c_new

        # Retorna prediccions + alphas acumulades (per a Doubly Stochastic loss)
        return torch.cat(preds, dim=0), alphas_sum

    @torch.no_grad()
    def beam_search(
        self,
        encoder_out: torch.Tensor,
        start_idx: int,
        end_idx: int,
        beam_size: int = 3,
    ) -> list[int]:
        """Beam search for a single image. Returns token ids (no special tokens)."""
        # encoder_out: [1, 49, 2048] → UNA sola imatge
        # start_idx: índex de <start>, end_idx: índex de <end>
        # beam_size=3 → mantenim els 3 millors camins en paral·lel
        device = encoder_out.device
        k = beam_size  # k = nombre de camins actius (comença a 3)

        enc = encoder_out.expand(k, -1, -1)               # [k, num_pixels, enc_dim] → replica la imatge k vegades
        h, c = self._init_hidden(enc)                      # [k, hidden_size] → un h i c per cada camí

        seqs = torch.full((k, 1), start_idx, dtype=torch.long, device=device)
        # [3, 1] → els 3 camins comencen amb <start>
        scores = torch.zeros(k, device=device)
        # [3] → puntuació acumulada de cada camí (log-prob), comença a 0

        complete_seqs, complete_scores = [], []
        # llistes on guardarem els camins que han acabat amb <end>

        for step in range(self.max_seq_length):
            embeddings = self.embed(seqs[:, -1])            # [k, embed_size] → embedding última paraula de cada camí
            context, _ = self.attention(enc, h)            # [k, enc_dim] → context d'atenció per cada camí
            lstm_in = torch.cat([embeddings, context], dim=1)  # [k, 2304]
            h, c = self.lstm_cell(lstm_in, (h, c))         # [k, 512] → nou estat

            log_probs = torch.log_softmax(self.fc(h), dim=1)  # [k, vocab_size] → log-probabilitats
            total = scores.unsqueeze(1) + log_probs            # [k, vocab_size] → score acumulat + nou

            if step == 0:
                top_scores, top_words = total[0].topk(k)
                # al primer pas tots els camins són iguals, agafem el top-k del primer
            else:
                top_scores, top_words = total.view(-1).topk(k)
                # aplana [k, 2982] → [k*2982] i agafa els k millors globalment

            beam_idx = top_words // self.vocab_size
            # a quin dels k camins pertany cada paraula seleccionada
            word_idx = top_words % self.vocab_size
            # quina paraula concreta és

            seqs = torch.cat([seqs[beam_idx], word_idx.unsqueeze(1)], dim=1)
            # actualitza les seqüències: afegeix la nova paraula a cada camí
            h, c = h[beam_idx], c[beam_idx]   # reorganitza estats segons beam_idx
            enc = enc[beam_idx]               # reorganitza imatge
            scores = top_scores               # actualitza puntuacions

            still_running = []
            for j in range(k):
                if word_idx[j].item() == end_idx:
                    complete_seqs.append(seqs[j, 1:-1].tolist())  # strip <start>/<end>
                    # guarda la seqüència sense <start> ni <end>
                    complete_scores.append(scores[j].item())
                else:
                    still_running.append(j)  # aquest camí encara no ha acabat

            if not still_running:
                break  # tots els camins han acabat, parem

            k = len(still_running)         # actualitza k (menys camins actius)
            seqs = seqs[still_running]
            h, c = h[still_running], c[still_running]
            enc = enc[still_running]
            scores = scores[still_running]

        if not complete_seqs:
            complete_seqs = [seqs[0, 1:].tolist()]
            complete_scores = [scores[0].item()]
            # si cap camí ha acabat amb <end>, agafa el millor que hi hagi

        best = max(range(len(complete_scores)), key=lambda i: complete_scores[i])
        return complete_seqs[best]
        # retorna la seqüència amb la puntuació (log-prob) més alta

    def sample_batch_with_logprobs(
        self,
        encoder_out: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 20,
    ):
        """Batched multinomial sampling for SCST — processes all B images in parallel.

        Returns (list[list[int]], list[Tensor]) — tokens and log_probs per image.
        log_probs retain the computation graph for REINFORCE.
        """
        B = encoder_out.size(0)
        device = encoder_out.device
        h, c = self._init_hidden(encoder_out)                  # [B, hidden_size]
        word = torch.full((B,), start_idx, dtype=torch.long, device=device)

        alive = torch.ones(B, dtype=torch.bool, device=device)
        tokens_per: list[list[int]] = [[] for _ in range(B)]
        lp_per:     list[list] = [[] for _ in range(B)]

        for _ in range(max_len):
            emb = self.embed(word)                             # [B, embed]
            context, _ = self.attention(encoder_out, h)       # [B, enc_dim]
            h, c = self.lstm_cell(torch.cat([emb, context], 1), (h, c))
            log_prob_dist = torch.log_softmax(self.fc(self.dropout(h)), 1)  # [B, V]
            word = torch.multinomial(log_prob_dist.exp(), 1).squeeze(1)     # [B]

            for i in range(B):
                if not alive[i]:
                    continue
                tok = word[i].item()
                if tok == end_idx:
                    alive[i] = False
                else:
                    tokens_per[i].append(tok)
                    lp_per[i].append(log_prob_dist[i, tok])

            if not alive.any():
                break

        log_probs_out = [
            torch.stack(lp) if lp else torch.zeros(1, device=device)
            for lp in lp_per
        ]
        return tokens_per, log_probs_out

    @torch.no_grad()
    def greedy_batch(
        self,
        encoder_out: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 20,
    ) -> list[list[int]]:
        """Batched greedy decode — SCST baseline, no gradients."""
        B = encoder_out.size(0)
        device = encoder_out.device
        h, c = self._init_hidden(encoder_out)
        word = torch.full((B,), start_idx, dtype=torch.long, device=device)

        alive = torch.ones(B, dtype=torch.bool, device=device)
        tokens_per: list[list[int]] = [[] for _ in range(B)]

        for _ in range(max_len):
            emb = self.embed(word)
            context, _ = self.attention(encoder_out, h)
            h, c = self.lstm_cell(torch.cat([emb, context], 1), (h, c))
            word = self.fc(h).argmax(1)                        # [B]

            for i in range(B):
                if not alive[i]:
                    continue
                tok = word[i].item()
                if tok == end_idx:
                    alive[i] = False
                else:
                    tokens_per[i].append(tok)

            if not alive.any():
                break

        return tokens_per

"""
📷 IMAGEN
[B, 3, H, W]
   ↓
🧱 CNN (ResNet-50/152 preentrenada, congelada)
   ↓
[B, 2048, 7, 7]
   ↓ reshape
[B, 49, 2048]
   ↓
49 regiones visuales (cada una = vector 2048)

   ↓
👀 ATTENTION (Bahdanau)

INPUT:
encoder_out [B, 49, 2048]
h_lstm      [B, 512]

→ proyección:
2048 → 256
512  → 256

→ score por región:
[B, 49, 1] → squeeze → [B, 49]

→ softmax:
alpha [B, 49]  (pesos que suman 1)

→ contexto:
weighted sum sobre regiones
context = [B, 2048]

   ↓
🧾 EMBEDDING PALABRA
token → [B]
→ embedding lookup
→ [B, 256]

   ↓
🔁 FUSIÓN
concat:
embedding [B, 256] + context [B, 2048]
→ [B, 2304]

   ↓
🧠 LSTM DECODER
LSTMCell:
input [B, 2304]
hidden → [B, 512]

   ↓
📚 OUTPUT VOCABULARIO
Linear:
[B, 512] → [B, vocab_size] (≈10000)

   ↓
Softmax:
probabilidades de palabras

   ↓
📝 PALABRA SIGUIENTE

(repetir autoregresivamente hasta <end>)"""
