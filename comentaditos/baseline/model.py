"""
model.py  —  Model Base (src/baseline/model.py)
================================================
CNN encoder + LSTM decoder per a image captioning (arquitectura "Show and Tell").

Basat en l'arquitectura original de Vinyals et al., 2015.

Aquesta és la versió SIMPLE del projecte: sense atenció.
L'encoder comprimeix tota la imatge en UN SOL VECTOR, i el decoder
genera la caption a partir d'aquest únic vector global.

Comparació amb el model d'atenció:
    Baseline:  imatge → 1 vector [B, 2048] → decoder genera tota la caption
    Atenció:   imatge → 49 vectors [B, 49, 2048] → decoder "mira" on cal en cada pas

Estructura:
    1. EncoderCNN  → ResNet preentrenada que extreu un vector de característiques global
    2. DecoderRNN  → LSTM que genera tokens un a un fins a <end> o max_seq_length
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
# pack_padded_sequence: compacta seqüències de longitud variable eliminant el padding,
# perquè la LSTM no processi tokens <pad> innecessàriament.


# ════════════════════════════════════════════════════════════════════
# 1. ENCODER CNN
# ════════════════════════════════════════════════════════════════════

class EncoderCNN(nn.Module):
    """ResNet preentrenada que converteix una imatge en un vector d'embedding fix.

    Diferència clau respecte a EncoderCNNAttention:
        Baseline  [:-1] → elimina NOMÉS la fc → avgpool comprimeix a [B, 2048, 1, 1]
                          → un sol vector global per imatge
        Atenció   [:-2] → elimina avgpool I fc → manté [B, 2048, 7, 7]
                          → 49 vectors regionals per imatge

    Per al baseline, aquest vector global s'injecta al decoder com a
    PRIMER INPUT de la LSTM (en lloc del token <start>).
    """

    def __init__(self, embed_size: int = 256, backbone: str = "resnet50"):
        """
        Args:
            embed_size: mida del vector de sortida (ha de coincidir amb embed_size del decoder)
                        Defecte 256. Si s'usen GloVe/Word2Vec, s'ajusta automàticament.
            backbone:   "resnet50" (25M params, ràpid) o "resnet152" (60M params, millors resultats)
        """
        super().__init__()
        # super().__init__() és OBLIGATORI per inicialitzar nn.Module.
        # Sense ell, .to(device), .parameters(), .train(), .eval() no funcionarien.

        # ── Carreguem la ResNet preentrenada ──────────────────────────────────
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            # IMAGENET1K_V2: versió millorada dels pesos d'ImageNet. La xarxa ja
            # sap detectar vores, textures, formes, parts d'objectes, animals, etc.
            net = models.resnet50(weights=weights)
        elif backbone == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V2
            net = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ── Eliminem l'última capa (la de classificació en 1000 classes) ─────
        # net.children() retorna les capes en ordre:
        # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # list()[:-1] elimina NOMÉS la fc (la capa de classificació dels 1000 ImageNet)
        # → mantenim avgpool, que comprimeix [B, 2048, 7, 7] → [B, 2048, 1, 1]
        modules = list(net.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        # nn.Sequential encadena les capes en ordre: la sortida d'una és l'entrada de la següent

        # ── Capa lineal de projecció ──────────────────────────────────────────
        self.linear = nn.Linear(net.fc.in_features, embed_size)
        # net.fc.in_features: mida de les característiques ABANS de la fc original
        #                     = 2048 per a ResNet-50 i ResNet-152
        # Converteix [B, 2048] → [B, embed_size]
        # Necessari perquè embed_size del decoder pot ser diferent de 2048.

        # ── Batch Normalization ────────────────────────────────────────────────
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        # Normalitza les activacions del batch: calcula la mitjana i desviació estàndard
        # del mini-batch actual i centra els valors al voltant de 0.
        # momentum=0.01: pes molt baix per a l'actualització de les estadístiques mòbils.
        # Un momentum baix fa que les estadístiques s'actualitzin molt lentament,
        # perquè les imatges d'un batch de captions no són independents (una imatge
        # pot aparèixer múltiples vegades amb captions diferents).

        # ── Congelació de la CNN ──────────────────────────────────────────────
        # NO entrenem els pesos de la ResNet. Les característiques que ha après
        # a ImageNet (vores, textures, objectes) ja ens serveixen tal com estan.
        # Entrenaríem MOLTS paràmetres sense necessitat i el model overfitting ràpidament.
        # Nota de futures millores: podríem descongelar la última capa (layer4)
        # per fer fine-tuning, que és una de les millores previstes.
        for p in self.cnn.parameters():
            p.requires_grad = False  # gradients = False → no s'actualitzen

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extreu el vector de característiques global d'un batch d'imatges.

        Args:
            images: [B, 3, 224, 224] — batch d'imatges normalitzades

        Returns:
            [B, embed_size] — un vector de 256 dimensions per imatge
        """
        with torch.no_grad():
            # Reforça que la CNN no calculi gradients.
            # Com que requires_grad=False ja ho garanteix, aquí és redundant,
            # però fa el codi més llegible i explícit.
            features = self.cnn(images)  # [B, 3, 224, 224] → [B, 2048, 1, 1]
            # Passa per totes les capes convolucionals + avgpool → [B, 2048, 1, 1]
            # La forma 1×1 indica que tota la informació espacial s'ha comprimit en un punt.

        features = features.flatten(1)
        # [B, 2048, 1, 1] → [B, 2048]
        # flatten(1) aplana totes les dimensions a partir de la 1 (deixa la dimensió de batch).

        features = self.bn(self.linear(features))
        # linear: [B, 2048] → [B, embed_size]  (projecció lineal)
        # bn:     [B, embed_size] → [B, embed_size] (normalització, valors prop de 0)
        # S'aplica bn SOBRE la sortida de linear, NO sobre les característiques crues.

        return features  # [B, embed_size] — vector global de la imatge


# ════════════════════════════════════════════════════════════════════
# 2. DECODER RNN (LSTM)
# ════════════════════════════════════════════════════════════════════

class DecoderRNN(nn.Module):
    """
    LSTM decoder que genera captions token per token, condicionat a les característiques de la imatge.

    Estratègia d'injecció de la imatge:
        A diferència del model amb atenció (que usa la imatge per inicialitzar h i c),
        aquí el vector de la imatge s'INJECTA com el PRIMER ELEMENT de la seqüència
        d'entrada a la LSTM, just davant del token <start>.
        Seqüència d'entrada a la LSTM: [features_imatge, emb(<start>), emb("a"), emb("dog"), ...]
        Seqüència de targets:                              ["a",         "dog",   <end>]

    Nota: usa nn.LSTM (no LSTMCell), que processa tota la seqüència alhora
    durant l'entrenament (molt eficient), i genera token per token durant la inferència.
    """

    def __init__(
        self,
        embed_size:         int,
        hidden_size:        int,
        vocab_size:         int,
        num_layers:         int = 1,
        max_seq_length:     int = 20,
        dropout:            float = 0.5,
        pretrained_weights: "torch.Tensor | None" = None,
        freeze_embeddings:  bool = False,
    ):
        """
        Args:
            embed_size:         mida dels embeddings de paraules (ha de coincidir amb embed_size de l'encoder)
            hidden_size:        mida de l'estat ocult de la LSTM (ex: 512)
            vocab_size:         nombre de paraules del vocabulari (~3000 per a Flickr8k)
            num_layers:         nombre de capes LSTM apilades (1 és suficient per a aquest problema)
            max_seq_length:     longitud màxima de les captions generades durant la inferència
            dropout:            probabilitat de dropout per regularització
            pretrained_weights: matriu [vocab_size, embed_size] de GloVe o Word2Vec (o None)
            freeze_embeddings:  si True, els embeddings NO s'actualitzen durant l'entrenament
        """
        super().__init__()

        # ── Capa d'embedding ──────────────────────────────────────────────────
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Taula de cerca: índex enter → vector d'embedding
        # Internament és una matriu [vocab_size, embed_size] on cada fila = un vector de paraula.
        # Exemple: self.embed(tensor([4, 7, 23])) → tensor [3, embed_size]

        if pretrained_weights is not None:
            # Si tenim GloVe o Word2Vec, els usem com a inicialització.
            # nn.Parameter fa que PyTorch tracti el tensor com a paràmetre del model
            # (apareixerà a model.parameters() i es guardarà al checkpoint).
            self.embed.weight = nn.Parameter(pretrained_weights)

        if freeze_embeddings:
            # Si volem provar si els embeddings preentrenats ja són prou bons
            # sense actualitzar-los, posem requires_grad=False.
            # En els experiments, el fine-tuning (freeze=False) ha donat millors resultats.
            self.embed.weight.requires_grad = False

        # ── Dropout ───────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        # Apaga aleatòriament 'dropout' fracció de les activacions durant l'entrenament.
        # Durant eval() (validació/test), el dropout s'inactiva automàticament.
        # Ajuda a evitar que el model memoritzi seqüències específiques del training set.

        # ── LSTM ──────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=embed_size,      # mida dels vectors d'entrada (embeddings)
            hidden_size=hidden_size,    # mida de l'estat ocult h (ex: 512)
            num_layers=num_layers,      # capes LSTM apilades (normalment 1)
            batch_first=True,           # les dimensions seran [B, T, ...] en lloc de [T, B, ...]
        )
        # nn.LSTM processa tota la seqüència T alhora (mode entrenament):
        #   input [B, T, embed_size] → output [B, T, hidden_size] + (h_n, c_n)
        # Durant la inferència (sample()), la cridem pas per pas amb seqüències de longitud 1.

        # ── Capa de sortida ────────────────────────────────────────────────────
        self.linear = nn.Linear(hidden_size, vocab_size)
        # Transforma l'estat ocult [hidden_size=512] en puntuacions per a cada paraula [vocab_size~3000].
        # Aplicant softmax sobre la sortida s'obté la distribució de probabilitat del vocabulari.

        self.max_seq_length = max_seq_length  # límit de tokens generats durant la inferència

    # ─── Forward (entrenament amb teacher forcing) ────────────────────────────

    def forward(
        self,
        features:  torch.Tensor,  # [B, embed_size] — vector global de la imatge (de l'encoder)
        captions:  torch.Tensor,  # [B, T] — tokens de les captions (inclou <start>)
        lengths:   list[int],     # longitud real de cada caption (inclou <start> i <end>)
    ):
        """
        Genera prediccions per a tots els tokens de les captions (mode entrenament).

        Usa TEACHER FORCING: l'input del pas t és sempre el token CORRECTE de la caption,
        no el token que el model va predir al pas anterior.
        Avantatge: entrenament molt més estable i ràpid.

        Seqüència d'entrada a la LSTM per a una caption ["a", "dog", <end>]:
            [features_imatge, emb(<start>), emb("a"), emb("dog")]  ← T+1 elements
        Targets esperats:
            [emb(<start>)→"a", emb("a")→"dog", emb("dog")→<end>]  ← T elements

        Returns:
            [sum(lengths), vocab_size] — prediccions compactades (sense padding)
        """
        embeddings = self.dropout(self.embed(captions))
        # captions [B, T] → embed → [B, T, embed_size] → dropout → [B, T, embed_size]

        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        # features [B, embed_size] → unsqueeze(1) → [B, 1, embed_size]
        # Concatenació al principi: [B, 1, embed_size] + [B, T, embed_size] → [B, T+1, embed_size]
        # La imatge es tracta com si fos el "token 0" de la seqüència.
        # Exemple: si la caption és [<start>, "a", "dog"] (T=3),
        # la seqüència resultant és [img_vec, <start>, "a", "dog"] (T+1=4)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # pack_padded_sequence elimina els tokens de padding del tensor i
        # els compacta en un format especial que la LSTM pot processar eficientment.
        # lengths ha d'estar en ordre DECREIXENT (ja ho garanteix collate_fn).
        # lengths de la seqüència [B, T+1] → [l1+1, l2+1, ...] (sumem 1 per la imatge)
        # Nota: en realitat, lengths és la longitud original de les captions
        # (inclou <start> i <end>), i pack_padded_sequence s'encarrega de la resta.

        hiddens, _ = self.lstm(packed)
        # Passa la seqüència compactada per la LSTM.
        # hiddens: tensor compactat amb els estats ocults per a cada token real (sense padding)
        # _ : estats finals (h_n, c_n) que no necessitem per a la loss

        outputs = self.linear(self.dropout(hiddens.data))
        # hiddens.data: descompacta el tensor compactat → [sum(lengths), hidden_size]
        #               tots els tokens de tots els exemples del batch apilats
        # dropout: [sum(lengths), hidden_size]
        # linear:  [sum(lengths), hidden_size] → [sum(lengths), vocab_size]
        # Cada fila és la predicció de probabilitats sobre tot el vocabulari per un token.

        return outputs

    # ─── Sample (inferència greedy) ───────────────────────────────────────────

    @torch.no_grad()
    def sample(self, features: torch.Tensor, states=None) -> torch.Tensor:
        """
        Genera captions per a un batch d'imatges usant decodificació greedy.

        Decodificació greedy: en cada pas, tria la paraula amb probabilitat MÉS ALTA.
        És la estratègia més senzilla i ràpida, però no garanteix la seqüència global òptima.
        (El model amb atenció usa beam search, que és millor però més lent.)

        El primer input és el vector de la imatge (features). A partir del pas 2,
        l'input és l'embedding de la paraula generada al pas anterior.

        Args:
            features: [B, embed_size] — vectors globals de les imatges
            states:   (h_0, c_0) estats inicials de la LSTM. Si None → zeros.

        Returns:
            [B, max_seq_length] — tensor d'índexs de tokens generats per cada imatge
        """
        sampled = []  # llista on guardarem els tokens generats a cada pas

        inputs = features.unsqueeze(1)
        # [B, embed_size] → [B, 1, embed_size]
        # La LSTM espera [B, T, input_size]. Aquí T=1 perquè processem un sol pas.
        # El primer input és el vector de la imatge (no l'embedding de <start>).

        for _ in range(self.max_seq_length):

            hiddens, states = self.lstm(inputs, states)
            # inputs [B, 1, embed_size] → hiddens [B, 1, hidden_size]
            # states s'actualitza: conté (h_t, c_t) per al pas següent.
            # La primera iteració: states=None → LSTM inicialitza a zeros.
            # Les iteracions següents: states conté la memòria acumulada.

            outputs = self.linear(hiddens.squeeze(1))
            # hiddens.squeeze(1): [B, 1, hidden_size] → [B, hidden_size]  (elimina la dimensió de seqüència)
            # linear:             [B, hidden_size]    → [B, vocab_size]
            # Puntuació per a cada paraula del vocabulari.

            _, predicted = outputs.max(1)
            # outputs.max(1): troba el valor màxim i el seu índex al llarg de la dimensió 1 (vocab)
            # _ = valor màxim (no el necessitem)
            # predicted = índex del màxim = la paraula amb major probabilitat [B]

            sampled.append(predicted)
            # Guardem el token generat en aquest pas

            inputs = self.embed(predicted).unsqueeze(1)
            # predicted [B] → embed → [B, embed_size] → unsqueeze(1) → [B, 1, embed_size]
            # El token generat es converteix en l'input del pas SEGÜENT.
            # (A diferència del forward(), aquí no hi ha teacher forcing.)

        return torch.stack(sampled, dim=1)
        # sampled és una llista de max_seq_length tensors [B]
        # torch.stack apila al llarg de dim=1: llista de [B] → [B, max_seq_length]
        # Cada fila és la seqüència de tokens generats per a una imatge.


# ════════════════════════════════════════════════════════════════════
# RESUM VISUAL DEL FLUX
# ════════════════════════════════════════════════════════════════════

# ENTRENAMENT (forward pass):
#
#    images [B, 3, 224, 224]
#        ↓  EncoderCNN
#    features [B, embed_size]
#        ↓  DecoderRNN.forward(features, captions, lengths)
#    outputs [sum(lengths), vocab_size]
#        ↓  CrossEntropyLoss vs targets
#    loss (escalar) → .backward() → optimizer.step()
#
#
# INFERÈNCIA (sample):
#
#    image [1, 3, 224, 224]  (o [B, ...] per a batch)
#        ↓  EncoderCNN
#    features [1, embed_size]
#        ↓  DecoderRNN.sample(features)
#    token ids [1, max_seq_length]
#        ↓  vocab.decode()
#    "a dog running on grass"
#
#
# DIMENSIONS EXEMPLE (batch_size=32, embed_size=256, hidden_size=512, vocab_size=2982):
#
#    EncoderCNN:
#        Input:             [32, 3, 224, 224]
#        Despres de ResNet: [32, 2048, 1, 1]
#        Despres flatten:   [32, 2048]
#        Despres linear+bn: [32, 256]
#
#    DecoderRNN (forward):
#        captions:          [32, 20]   (T=20 tokens max)
#        Despres embed:     [32, 20, 256]
#        Cat amb features:  [32, 21, 256]  (+1 per la imatge al davant)
#        Despres LSTM:      [sum(lengths), 512]  (compactat, sense padding)
#        Despres linear:    [sum(lengths), 2982]