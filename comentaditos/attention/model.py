"""
model.py  —  Model amb Atenció (src/attention/model.py)
=======================================================
CNN encoder espacial + LSTM decoder amb atenció de Bahdanau + beam search.

Basat en: "Show, Attend and Tell: Neural Image Caption Generation
           with Visual Attention" (Xu et al., 2015)

Diferència clau respecte al baseline:
    Baseline:  una sola imatge → un sol vector global [B, 2048] → decoder
    Atenció:   una sola imatge → 49 vectors regionals [B, 49, 2048]
               En cada pas de la generació, el decoder "mira" les regions
               rellevants per a la paraula que vol generar.

Estructura:
    1. EncoderCNNAttention  → extreu la graella de característiques 7×7
    2. Attention (Bahdanau) → calcula pesos d'atenció sobre les 49 regions
    3. AttentionDecoder     → genera la caption amb LSTMCell + atenció
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


# ════════════════════════════════════════════════════════════════════
# 1. ENCODER CNN AMB ATENCIÓ
# ════════════════════════════════════════════════════════════════════

class EncoderCNNAttention(nn.Module):
    """ResNet que retorna una graella espacial de vectors en lloc d'un únic vector global.

    Baseline:   elimina les 2 últimes capes (avgpool + fc) → [B, 2048, 1, 1]
                                                              (mapa de 1×1 → un sol punt)
    Atenció:    elimina NOMÉS la última capa (fc) → NO, elimina avgpool I fc → [B, 2048, 7, 7]
                Millor dit: elimina avgpool i fc → obté el mapa espacial complet.

    Per a una imatge 224×224:
        La ResNet-50 redueix progressivament la resolució:
        224×224 → 112×112 → 56×56 → 28×28 → 14×14 → 7×7
        Al darrer bloc convolucional queden 7×7 = 49 regions, cadascuna
        amb 2048 valors. Cada region "representa" un 32×32 tros de la imatge original.
    """

    def __init__(self, backbone: str = "resnet50"):
        """
        Args:
            backbone: arquitectura ResNet a usar.
                      "resnet50"  → 25M paràmetres, més ràpid
                      "resnet152" → 60M paràmetres, millors resultats (usat als experiments)
        """
        super().__init__()

        # ── Carreguem la ResNet preentrenada ──────────────────────────────────
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            # IMAGENET1K_V2 és la versió millorada dels pesos preentrenats d'ImageNet
            net = models.resnet50(weights=weights)
        elif backbone == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V2
            net = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ── Eliminem les dues últimes capes ───────────────────────────────────
        # net.children() retorna les capes de la ResNet com a generador:
        # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # list()[:-2] elimina les dues últimes (avgpool i fc)
        # → ens quedem amb tot fins a layer4 inclòs
        # → la sortida és [B, 2048, 7, 7] (mapa espacial de 7×7 regions)
        self.cnn = nn.Sequential(*list(net.children())[:-2])

        # Diferència visual:
        # Baseline [:-1] → elimina només fc → sortiria [B, 2048, 1, 1] (avgpool comprimeix tot)
        # Atenció  [:-2] → elimina avgpool i fc → [B, 2048, 7, 7] (mantenim la graella espacial)

        self.encoder_dim = 2048  # dimensió de cada vector de regió (fixada per ResNet)

        # ── Congelació de la CNN ──────────────────────────────────────────────
        # No entrenem els pesos de la ResNet. Fem servir les característiques
        # que ja va aprendre amb ImageNet (vores, textures, objectes, etc.).
        # Avantatge: menys paràmetres a entrenar → més ràpid i menys overfitting.
        # En futures iteracions es podria descongelar la última capa de la ResNet
        # per fer fine-tuning (és una de les millores previstes).
        for p in self.cnn.parameters():
            p.requires_grad = False  # no calcular gradients → no actualitzar pesos

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Processa un batch d'imatges i retorna les 49 regions de característiques.

        Args:
            images: [B, 3, 224, 224] — batch d'imatges normalitzades

        Returns:
            [B, 49, 2048] — cada imatge representada com 49 vectors de 2048 dimensions
        """
        with torch.no_grad():
            # torch.no_grad() reforça que no es calculen gradients aquí.
            # Com que la CNN és congelada (requires_grad=False), és redundant
            # però fa explícit que aquest còmput no participa en el backprop.
            out = self.cnn(images)  # [B, 3, 224, 224] → [B, 2048, 7, 7]

        B, C, H, W = out.shape
        # B = batch size, C = 2048 canals, H = 7 alçada, W = 7 amplada

        out = out.permute(0, 2, 3, 1)
        # Canvia l'ordre de les dimensions: [B, C, H, W] → [B, H, W, C]
        # = [B, 7, 7, 2048]
        # Movem els canals al final perquè el mòdul d'atenció espera [B, regions, dim]

        return out.view(B, H * W, C)
        # Aplanem la graella 7×7 en una seqüència lineal de 49 regions:
        # [B, 7, 7, 2048] → [B, 49, 2048]
        # Cada una de les 49 files representa una regió de la imatge original.


# ════════════════════════════════════════════════════════════════════
# 2. MÒDUL D'ATENCIÓ (Bahdanau / Additiva)
# ════════════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """Atenció additiva de Bahdanau sobre les 49 regions espacials de l'encoder.

    La intuïció:
        En cada pas de la generació (ex: volem generar la paraula "dog"),
        el decoder ha de saber a QUINA PART de la imatge mirar.
        L'atenció calcula automàticament un pes (alpha) per a cada una de les
        49 regions: la que conté el gos rebrà un alpha alt (~0.8), el cel un alpha
        baix (~0.01), etc.

    Mecànica (Bahdanau, 2015):
        energy[i] = W · tanh(U · region[i] + V · hidden_state)
        alpha      = softmax(energy)           ← pesos d'atenció, sumen 1
        context    = sum_i(alpha[i] · region[i])  ← vector de context ponderat

    Notació de dimensions:
        B          = batch size
        num_pixels = 49  (7×7 regions)
        encoder_dim   = 2048
        decoder_dim   = 512  (mida de l'estat ocult h de la LSTM)
        attention_dim = 256  (dimensió interna del mòdul d'atenció)
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        Args:
            encoder_dim:   dimensió dels vectors de les regions (2048)
            decoder_dim:   dimensió de l'estat ocult de la LSTM (512)
            attention_dim: dimensió del espai d'atenció intern (256)
        """
        super().__init__()

        # Capa lineal que projecta les regions de l'encoder a l'espai d'atenció.
        # 2048 → 256. Aplicada a totes les 49 regions alhora.
        self.enc_att  = nn.Linear(encoder_dim, attention_dim)

        # Capa lineal que projecta l'estat ocult del decoder a l'espai d'atenció.
        # 512 → 256. S'aplica al vector h del pas actual.
        self.dec_att  = nn.Linear(decoder_dim, attention_dim)

        # Capa lineal que calcula un score escalar per a cada regió.
        # 256 → 1. Un sol número per regió indica la seva "importància".
        self.full_att = nn.Linear(attention_dim, 1)

        # Softmax per normalitzar els 49 scores de les regions en pesos que sumen 1.
        # dim=1 vol dir que el softmax s'aplica al llarg de la dimensió de les regions (49).
        self.softmax  = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, h: torch.Tensor):
        """Calcula el vector de context i els pesos d'atenció per al pas actual.

        Args:
            encoder_out: [B, 49, 2048] — les 49 regions de la imatge
            h:           [B, 512]      — l'estat ocult actual del decoder

        Returns:
            context: [B, 2048] — resum ponderat de la imatge per a aquest pas
            alpha:   [B, 49]   — pesos d'atenció (un per cada regió, sumen 1)
        """

        # ── Càlcul dels energy scores ─────────────────────────────────────────

        enc_proj = self.enc_att(encoder_out)
        # Projecta cada una de les 49 regions: [B, 49, 2048] → [B, 49, 256]
        # Captura la informació de CONTINGUT de cada regió.

        dec_proj = self.dec_att(h).unsqueeze(1)
        # Projecta l'estat del decoder: [B, 512] → [B, 256] → [B, 1, 256]
        # .unsqueeze(1) afegeix la dimensió de regions (1) per poder sumar amb enc_proj.
        # Captura "QUÈ ESTEM BUSCANT" en aquest moment de la generació.

        combined = torch.tanh(enc_proj + dec_proj)
        # Suma les dues projeccions (broadcasting: [B,49,256] + [B,1,256] → [B,49,256]).
        # tanh combina i satura els valors entre -1 i 1.
        # Resultat: [B, 49, 256] — informació combinada de cada regió + estat actual.

        e = self.full_att(combined).squeeze(2)
        # full_att projecta de 256 → 1: [B, 49, 256] → [B, 49, 1]
        # .squeeze(2) elimina la última dimensió: [B, 49, 1] → [B, 49]
        # e[b][i] = score (importància) de la regió i per a la mostra b

        # ── Normalització amb Softmax ─────────────────────────────────────────
        alpha = self.softmax(e)  # [B, 49] — els 49 scores es converteixen en pesos
        # Cada fila suma 1.0: alpha[b][0] + alpha[b][1] + ... + alpha[b][48] = 1.0
        # Exemple: alpha[b] = [0.01, 0.03, ..., 0.82, ..., 0.01]  ← region del gos = 0.82

        # ── Vector de context ─────────────────────────────────────────────────
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        # alpha.unsqueeze(2): [B, 49] → [B, 49, 1] (per poder multiplicar per encoder_out [B,49,2048])
        # encoder_out * alpha → [B, 49, 2048]: cada region ponderada pel seu pes
        # .sum(dim=1): suma les 49 regions → [B, 2048]
        # context és el "resum" de la imatge ponderat per l'atenció:
        # les regions importants contribueixen molt, les irrellevants quasi res.

        return context, alpha


# ════════════════════════════════════════════════════════════════════
# 3. DECODER AMB ATENCIÓ
# ════════════════════════════════════════════════════════════════════

class AttentionDecoder(nn.Module):
    """LSTM decoder amb atenció. Suporta teacher-forcing (train) i beam search (inferència).

    Diferències clau respecte al DecoderRNN (baseline):
        - Usa LSTMCell (un pas a la vegada) en lloc de LSTM (tota la seqüència alhora)
          perquè l'atenció s'ha de recalcular en cada pas.
        - La imatge NO s'injecta com a primer token, sinó que s'usa per inicialitzar
          els estats h i c de la LSTM (_init_hidden).
        - En cada pas, l'input de la LSTM és: [embedding de la paraula] + [context d'atenció]
          en lloc de només l'embedding. Per això l'input_size = embed_size + encoder_dim.
    """

    def __init__(
        self,
        encoder_dim:        int = 2048,   # mida dels vectors de les regions (de l'encoder)
        embed_size:         int = 256,    # mida dels vectors d'embedding de les paraules
        hidden_size:        int = 512,    # mida de l'estat ocult h de la LSTMCell
        vocab_size:         int = 10000,  # nombre de paraules del vocabulari
        attention_dim:      int = 256,    # dimensió interna del mòdul d'atenció
        dropout:            float = 0.5,  # probabilitat de desactivar neurones (regularització)
        max_seq_length:     int = 20,     # longitud màxima de la caption generada
        pretrained_weights: "torch.Tensor | None" = None,  # pesos GloVe/W2V [vocab_size, embed_size]
        freeze_embeddings:  bool = False, # si True, els embeddings no s'actualitzen durant train
    ):
        super().__init__()

        # Guardem els hiperparàmetres com a atributs (els necessite al beam search)
        self.encoder_dim    = encoder_dim
        self.hidden_size    = hidden_size
        self.vocab_size     = vocab_size
        self.max_seq_length = max_seq_length

        # ── Mòdul d'atenció ────────────────────────────────────────────────────
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        # Es crea una instancia de la classe Attention definida més amunt.

        # ── Capa d'embedding ──────────────────────────────────────────────────
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Taula de consulta: índex enter → vector d'embedding
        # Internament és una matriu [vocab_size, embed_size] on cada fila és una paraula.

        if pretrained_weights is not None:
            # Substituïm la inicialització aleatòria per pesos GloVe o Word2Vec.
            # nn.Parameter fa que PyTorch tracti aquesta matriu com un paràmetre
            # entrenable (a menys que es congeli amb freeze_embeddings).
            self.embed.weight = nn.Parameter(pretrained_weights)

        if freeze_embeddings:
            # Si freeze_embeddings=True, els pesos d'embedding NO s'actualitzen.
            # Útil per a tests on volem saber si els embeddings preentrenats solen
            # ser suficients sense actualitzar-los.
            # En els nostres experiments, el fine-tuning (freeze=False) ha demostrat
            # ser millor: els embeddings s'adapten al vocabulari específic de Flickr8k.
            self.embed.weight.requires_grad = False

        # ── Dropout ────────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        # Durant l'entrenament, apaga aleatòriament una fracció (dropout) de les
        # neurones en cada pas. Evita que el model memorize patrons específics
        # del training set (overfitting).
        # Durant eval() (validació/test), el dropout s'INACTIVA automàticament.

        # ── LSTMCell ───────────────────────────────────────────────────────────
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        # LSTMCell (NO LSTM!) processa UN SOL PAS de la seqüència a la vegada.
        # Usem LSTMCell perquè en cada pas hem de recalcular l'atenció.
        #
        # input_size  = embed_size + encoder_dim = 256 + 2048 = 2304
        #               L'input és la concatenació de:
        #                 - embedding de la paraula actual  [embed_size = 256]
        #                 - vector de context d'atenció     [encoder_dim = 2048]
        # output_size = hidden_size = 512  (mida de l'estat ocult h)

        # ── Capes d'inicialització de la LSTM ──────────────────────────────────
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        # Dues capes lineals per inicialitzar h i c de la LSTM a partir de la imatge.
        # [2048] → [512] per a h, [2048] → [512] per a c.
        # S'apliquen sobre la mitjana de les 49 regions (representació global de la imatge).
        # En el baseline, la imatge s'injectava com a primer token;
        # aquí s'usa per "posar en context" la LSTM des del principi.

        # ── Capa de sortida ────────────────────────────────────────────────────
        self.fc = nn.Linear(hidden_size, vocab_size)
        # Capa lineal final: [512] → [vocab_size (~3000)]
        # Transforma l'estat ocult de la LSTM en una puntuació per a cada paraula.
        # Aplicant softmax sobre aquesta sortida s'obté la distribució de probabilitat.

    # ─── Inicialització de la LSTM ─────────────────────────────────────────────

    def _init_hidden(self, encoder_out: torch.Tensor):
        """Inicialitza els estats h i c de la LSTM a partir de la imatge.

        Args:
            encoder_out: [B, 49, 2048] — les 49 regions de la imatge

        Returns:
            h: [B, 512] — estat ocult inicial
            c: [B, 512] — estat de cel·la inicial
        """
        mean = encoder_out.mean(dim=1)
        # Promig sobre les 49 regions: [B, 49, 2048] → [B, 2048]
        # Dona una representació GLOBAL de la imatge (sense atendre cap regió en particular)
        # que serveix per "orientar" la LSTM des del primer pas.

        h = torch.tanh(self.init_h(mean))  # [B, 2048] → [B, 512], tanh satura entre -1 i 1
        c = torch.tanh(self.init_c(mean))  # ídem per a l'estat de cel·la
        return h, c

    # ─── Forward (entrenament amb teacher forcing) ─────────────────────────────

    def forward(
        self,
        encoder_out: torch.Tensor,   # [B, 49, 2048]
        captions:    torch.Tensor,   # [B, T]  (tokens de les captions, inclou <start>)
        lengths:     list[int],      # longitud real de cada caption (inclou <start> i <end>)
    ):
        """Genera prediccions per a totes les posicions de les captions (mode entrenament).

        Usa TEACHER FORCING: en lloc d'usar la paraula generada al pas anterior
        com a input del pas actual, usa la paraula CORRECTA de la caption.
        Avantatge: l'entrenament és molt més estable i ràpid.
        Desavantatge: durant la inferència no hi ha teacher forcing, el que pot
        causar "exposure bias" (el model no ha vist els seus propis errors).

        Returns:
            preds concatenades: [sum(lengths-1), vocab_size]
            sum(lengths-1) = total de tokens a predir (no s'inclou <end> com a input)
        """
        B          = encoder_out.size(0)
        embeddings = self.dropout(self.embed(captions))
        # captions: [B, T] → embed: [B, T, embed_size] → dropout: [B, T, embed_size]

        h, c = self._init_hidden(encoder_out)
        # Inicialitzem h i c a partir de la imatge (no a zero com seria per defecte)

        # decode_lengths: longitud real - 1 perquè no cal predir res DESPRÉS de <end>
        # Ex: caption = [<start>, "a", "dog", <end>] → length=4 → decode_length=3
        # Hem de predir "a" (a partir de <start>), "dog" (a partir de "a"), <end> (a partir de "dog")
        decode_lengths = [l - 1 for l in lengths]
        max_t = max(decode_lengths)  # longitud màxima de decodificació al batch

        preds = []  # llista on guardem les prediccions de cada pas

        for t in range(max_t):
            # bt = quantes captions encara no han acabat en el pas t
            # Les captions estan ordenades de més llarga a més curta (gràcies a collate_fn)
            # per tant les primeres bt captions del batch encara continuen al pas t.
            bt = sum(1 for l in decode_lengths if l > t)

            # ── Atenció ──────────────────────────────────────────────────────
            context, _ = self.attention(encoder_out[:bt], h[:bt])
            # Calculem on "mira" el model en aquest pas, per als bt exemples actius.
            # encoder_out[:bt] → [bt, 49, 2048]
            # h[:bt]           → [bt, 512]
            # context          → [bt, 2048]

            # ── Input de la LSTM ─────────────────────────────────────────────
            lstm_in = torch.cat([embeddings[:bt, t], context], dim=1)
            # embeddings[:bt, t]: l'embedding de la paraula actual (pas t) per als bt exemples
            #                     [bt, embed_size = 256]
            # context:            el vector de context d'atenció
            #                     [bt, encoder_dim = 2048]
            # Concatenació:       [bt, 256 + 2048] = [bt, 2304]

            # ── Un pas de la LSTMCell ────────────────────────────────────────
            h_new, c_new = self.lstm_cell(lstm_in, (h[:bt], c[:bt]))
            # LSTMCell processa un sol pas i actualitza h i c.
            # h_new, c_new: [bt, 512]

            preds.append(self.fc(self.dropout(h_new)))
            # dropout sobre h_new → fc transforma [bt, 512] → [bt, vocab_size]
            # Predicció de la paraula del pas t per als bt exemples actius.

            # ── Actualitzem h i c per al pas següent ─────────────────────────
            if bt < B:
                # Algunes captions han acabat (les de longitud < t+1).
                # Les captions actives (0..bt-1) reben el nou h i c.
                # Les captions inactives (bt..B-1) mantenen l'estat anterior.
                h = torch.cat([h_new, h[bt:]], dim=0)
                c = torch.cat([c_new, c[bt:]], dim=0)
            else:
                # Totes les captions estan actives → actualitzem tot el batch
                h, c = h_new, c_new

        # Concatenem totes les prediccions de tots els passos en un únic tensor.
        # preds és una llista de tensors [bt, vocab_size] (bt pot variar per pas).
        # torch.cat concatena al llarg de dim=0 → [sum(decode_lengths), vocab_size]
        return torch.cat(preds, dim=0)

    # ─── Beam Search (inferència) ──────────────────────────────────────────────

    @torch.no_grad()  # No calculem gradients durant la inferència (estalvia memòria i temps)
    def beam_search(
        self,
        encoder_out: torch.Tensor,  # [1, 49, 2048] — UNA sola imatge
        start_idx:   int,           # índex de <start> al vocabulari
        end_idx:     int,           # índex de <end> al vocabulari
        beam_size:   int = 3,       # nombre de camins a mantenir en paral·lel
    ) -> list[int]:
        """Genera la caption per a una imatge usant beam search.

        Greedy vs Beam Search:
            Greedy:      en cada pas, tria la paraula amb probabilitat màxima.
                         → Ràpid, però pot perdre seqüències millors globalment.
            Beam Search: manté els 'beam_size' millors camins en paral·lel.
                         En cada pas, expandeix tots els camins i queda amb els top-k.
                         → Millors resultats, però més costós computacionalment.

        Exemple amb beam_size=3:
            Pas 0: 3 camins iguals → ["<start>"]
            Pas 1: expandim → 3 × vocab_size opcions → agafem top-3:
                   → "a dog..."  (score: -0.5)
                   → "two men..." (score: -0.7)
                   → "a man..."  (score: -0.8)
            Pas 2: expandim els 3 camins actius → top-3 globals
            ...
            Resultat: el camí complet amb la puntuació total (log-prob) màxima.

        Returns:
            llista d'índexs de tokens (sense <start> ni <end>)
        """
        device = encoder_out.device
        k      = beam_size  # nombre de camins actius (es redueix quan algun acaba)

        # ── Inicialització ────────────────────────────────────────────────────

        enc = encoder_out.expand(k, -1, -1)
        # Repliquem la imatge k vegades per tenir una còpia per a cada camí.
        # [1, 49, 2048] → [k, 49, 2048]  (-1 vol dir "no canviïs aquesta dimensió")

        h, c = self._init_hidden(enc)
        # Inicialitzem h i c per als k camins: [k, 512]

        seqs = torch.full((k, 1), start_idx, dtype=torch.long, device=device)
        # Tensor [k, 1] inicialitzat amb l'índex de <start>.
        # Cada camí comença amb la mateixa seqüència: [<start>]

        scores = torch.zeros(k, device=device)
        # Puntuació acumulada de log-probabilitats de cada camí.
        # Comença a 0 (log(1) = 0).

        complete_seqs   = []  # camins que han generat <end>
        complete_scores = []  # puntuació de cada camí complet

        # ── Bucle de generació ────────────────────────────────────────────────

        for step in range(self.max_seq_length):

            embeddings = self.embed(seqs[:, -1])
            # seqs[:, -1]: última paraula de cada camí [k]
            # embed: [k] → [k, embed_size]
            # És la input del pas actual de cada camí.

            context, _ = self.attention(enc, h)
            # Atenció per als k camins: [k, 2048]

            lstm_in = torch.cat([embeddings, context], dim=1)
            # [k, 256] + [k, 2048] → [k, 2304]

            h, c = self.lstm_cell(lstm_in, (h, c))
            # Un pas de LSTMCell per als k camins: h, c → [k, 512]

            log_probs = torch.log_softmax(self.fc(h), dim=1)
            # fc: [k, 512] → [k, vocab_size]
            # log_softmax: converteix logits en log-probabilitats [k, vocab_size]
            # Usem log-probabilitats perquè sumen en lloc de multiplicar (numèricament estable).

            total = scores.unsqueeze(1) + log_probs
            # scores: [k] → [k, 1] (unsqueeze per broadcasting)
            # scores[i] + log_probs[i][j] = puntuació total de "camí i continuat amb paraula j"
            # total: [k, vocab_size]

            # ── Selecció dels millors camins ──────────────────────────────────

            if step == 0:
                top_scores, top_words = total[0].topk(k)
                # Al primer pas tots els k camins són idèntics → mirem només el primer.
                # Agafem els k tokens amb major puntuació.
                # top_scores, top_words: [k]
            else:
                top_scores, top_words = total.view(-1).topk(k)
                # Aplanem tota la matriu [k, vocab_size] → [k*vocab_size]
                # Agafem globalment els k millors (paraula, camí) entre TOTES les combinacions.

            beam_idx = top_words // self.vocab_size
            # A quin dels k camins pertany cada selecció?
            # top_words és un índex pla en [k*vocab_size]:
            # dividir per vocab_size dóna el número de camí (0..k-1)

            word_idx = top_words % self.vocab_size
            # Quina paraula concreta és?
            # El residu de la divisió dóna la posició al vocabulari.

            seqs = torch.cat([seqs[beam_idx], word_idx.unsqueeze(1)], dim=1)
            # Reordenem les seqüències segons els camins seleccionats i afegim la nova paraula.
            # seqs[beam_idx]: [k, len_actual]  (reorganitzat per beam_idx)
            # word_idx.unsqueeze(1): [k, 1]
            # Resultat: [k, len_actual + 1]

            h, c = h[beam_idx], c[beam_idx]
            # Reordenem h i c per reflectir quins camins han estat seleccionats.

            enc = enc[beam_idx]
            # Reordenem la imatge (cada camí "porta" la seva còpia de l'encoder).

            scores = top_scores
            # Actualitzem les puntuacions acumulades.

            # ── Detecció de camins acabats ────────────────────────────────────

            still_running = []
            for j in range(k):
                if word_idx[j].item() == end_idx:
                    # Aquest camí ha generat <end> → el guardem com a complet.
                    # seqs[j, 1:-1]: elimina <start> (posició 0) i <end> (última posició)
                    complete_seqs.append(seqs[j, 1:-1].tolist())
                    complete_scores.append(scores[j].item())
                else:
                    still_running.append(j)  # aquest camí continua

            if not still_running:
                break  # tots els k camins han acabat

            # ── Actualitzem els camins actius ─────────────────────────────────
            k    = len(still_running)
            seqs = seqs[still_running]
            h, c = h[still_running], c[still_running]
            enc  = enc[still_running]
            scores = scores[still_running]

        # ── Resultat final ────────────────────────────────────────────────────

        if not complete_seqs:
            # Cap camí ha generat <end> (seqüència massa llarga o model mal entrenat).
            # Retornem el millor camí actiu (sense el <start>).
            complete_seqs   = [seqs[0, 1:].tolist()]
            complete_scores = [scores[0].item()]

        # Entre tots els camins complets, retornem el de puntuació (log-prob) màxima.
        best = max(range(len(complete_scores)), key=lambda i: complete_scores[i])
        return complete_seqs[best]


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
