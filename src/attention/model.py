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
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            net = models.resnet50(weights=weights)       # carrega ResNet-50 preentrenada
        elif backbone == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V2
            net = models.resnet152(weights=weights)
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():           # no calcula gradients (CNN congelada)
            out = self.cnn(images)            # [B, 2048, 7, 7] → mapa de 7x7 regions
        B, C, H, W = out.shape          # desempaqueta dimensions
        out = out.permute(0, 2, 3, 1)         # [B, 7, 7, 2048] → mou canals al final
        return out.view(B, H * W, C)          # [B, 49, 2048] → aplana la graella 7x7 en 49 regions
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
        embeddings = self.dropout(self.embed(captions))   # [B, T, embed_size]
        h, c = self._init_hidden(encoder_out)  # inicialitza LSTM amb info de la imatge

        decode_lengths = [l - 1 for l in lengths]
        # -1 perquè no cal predir res després de <end>
        max_t = max(decode_lengths)  # màxim de passos a fer

        preds = []
        for t in range(max_t):
            bt = sum(1 for l in decode_lengths if l > t)
            # bt = quantes captions encara no han acabat en el pas t
            # (les captions estan ordenades de més llarga a més curta)
            context, _ = self.attention(encoder_out[:bt], h[:bt])
            # calcula el context d'atenció per les bt captions actives → [bt, 2048]
            lstm_in = torch.cat([embeddings[:bt, t], context], dim=1)
            # concatena embedding de la paraula actual + context de la imatge
            # [bt, 256] + [bt, 2048] → [bt, 2304]
            h_new, c_new = self.lstm_cell(lstm_in, (h[:bt], c[:bt]))
            # un pas de la LSTM → nous estats h i c [bt, 512]
            preds.append(self.fc(self.dropout(h_new)))    # [bt, vocab_size]
            # dropout + capa lineal → predicció de la següent paraula
            if bt < B:
                h = torch.cat([h_new, h[bt:]], dim=0)
                c = torch.cat([c_new, c[bt:]], dim=0)
                # actualitza h i c: les captions actives amb el nou estat,
                # les que ja han acabat mantenen l'estat anterior
            else:
                h, c = h_new, c_new

        # Concatenation order matches pack_padded_sequence output layout
        return torch.cat(preds, dim=0)
        # concatena totes les prediccions → [sum(lengths-1), 2982]

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
