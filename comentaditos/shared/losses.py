"""
losses.py
=========
Funcions de pèrdua (loss functions) per a image captioning.

Què és una loss function?
    Durant l'entrenament, el model fa una predicció i la comparem amb la
    resposta correcta. La loss mesura "quant equivocat" ha estat el model.
    Quan la loss és alta, el model ha fallat molt. Quan és baixa, ha encertat.
    L'optimitzador ajusta els pesos del model per MINIMITZAR la loss.

Per què necessitem una loss personalitzada?
    La Cross-Entropy estàndard tracta totes les paraules incorrectes com a
    igual de "dolentes". Si la paraula correcta és "mountain" i el model prediu
    "hill" (pujol) o "cat" (gat), la penalització és la MATEIXA.
    Però "hill" és semànticament molt més propera a "mountain" que "cat".

    La SemanticCrossEntropyLoss soluciona això: penalitza menys les paraules
    semànticament similars al target, usant els embeddings GloVe/Word2Vec
    per saber quines paraules s'assemblen.

    Nota dels experiments: en la iteració actual, la SemanticCrossEntropyLoss
    ha resultat distorsionar la distribució de probabilitats (la perplexitat
    es dispara i el BLEU-4 baixa). La Cross-Entropy estàndard ha demostrat
    ser més estable. Deixem la implementació per a futures iteracions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓ DE LES SOFT LABELS
# ─────────────────────────────────────────────────────────────────────────────

def build_soft_labels(embed_weights: torch.Tensor, temperature: float = 10.0) -> torch.Tensor:
    """
    Construeix una matriu de soft labels semàntics a partir de vectors d'embedding.

    Concepte clau — SOFT LABELS vs HARD LABELS:
        Hard label (Cross-Entropy estàndard):
            Target "mountain" → [0, 0, 0, ..., 1, ..., 0]   (tot a zero excepte la paraula correcta)
            Predicció "hill"  → error màxim, igual que "cat"

        Soft label (SemanticCrossEntropyLoss):
            Target "mountain" → [0, 0, ..., 0.65, ..., 0.15, ..., 0.08, ...]
                                             ↑mountain      ↑hill       ↑peak
            Predicció "hill"  → error petit  (penalitza menys, és semànticament similar)
            Predicció "cat"   → error gran   (penalitza molt, no té res a veure)

    Com es calculen les soft labels?
        1. Per a cada parella de paraules (i, j), calculem la SIMILITUD COSINUS
           entre els seus vectors d'embedding.
           sim_cos(u, v) = (u·v) / (|u|·|v|)
           → 1.0 si els vectors apunten en la mateixa direcció (paraules molt similars)
           → 0.0 si els vectors són perpendiculars
           → negatius si apunten en direccions oposades

        2. Apliquem SOFTMAX escalat per 'temperature' per convertir les similituds
           en una distribució de probabilitat que suma 1.

    Paràmetre temperature:
        - Alta (ex. 10.0) → distribució concentrada al voltant del target
          (comportament semblant a la Cross-Entropy estàndard)
        - Baixa (ex. 1.0) → distribució molt uniforme, penalitza poc qualsevol paraula incorrecta
          (potencialment massa permissiva)
        Recomanat: 10.0 per a GloVe/Word2Vec

    Args:
        embed_weights: matriu d'embeddings [vocab_size, embed_dim]
                       Cada fila és el vector d'una paraula del vocabulari.
        temperature:   factor d'escala de les similituds cosinus (defecte 10.0)

    Returns:
        soft_labels: matriu [vocab_size, vocab_size]
                     Cada fila i és la distribució de probabilitat per al target i.
                     soft_labels[i][j] = quanta "atenció" mereix la paraula j quan la paraula target és i.
    """
    # Pas 1: Normalitzem els vectors a norma 1 per calcular similitud cosinus.
    # La similitud cosinus és: sim(u,v) = (u·v) / (|u|·|v|)
    # Si els vectors ja estan normalitzats (|u|=|v|=1): sim(u,v) = u·v
    # Podem calcular tota la matriu de similituds amb una multiplicació de matrius.

    norms = embed_weights.norm(dim=1, keepdim=True).clamp(min=1e-8)
    # .norm(dim=1) → longitud de cada vector → [vocab_size, 1]
    # .clamp(min=1e-8) → evitem divisió per zero

    normalized = embed_weights / norms
    # [vocab_size, embed_dim] → cada fila té norma 1.0

    # Pas 2: Matriu de similituds cosinus completa.
    # normalized @ normalized.T = tots els productes escalars entre parelles
    # Resultat [i][j] = similitud cosinus entre paraula i i paraula j, en [-1, 1]
    sim = normalized @ normalized.T  # [vocab_size, vocab_size]

    # Pas 3: Softmax escalat per temperatura → distribució de probabilitat per fila
    # Multiplicar per temperature exagera les diferències de similitud:
    # similituds altes guanyen molt de pes, les baixes en perden.
    return torch.softmax(sim * temperature, dim=1)
    # dim=1 → softmax per fila: cada fila suma 1.0


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE DE LOSS SEMÀNTICA
# ─────────────────────────────────────────────────────────────────────────────

class SemanticCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy amb soft labels semàntics basats en similitud d'embeddings.

    Hereda de nn.Module per poder moure la loss a GPU (.to(device)) i integrar-se
    amb el bucle d'entrenament estàndard de PyTorch.

    Fórmula:
        L = -∑_j  soft_label[target][j] · log(softmax(logits)[j])

        Si soft_label fos one-hot → equivalent a Cross-Entropy estàndard.
        Si soft_label és suau → paraules similars al target reben menys penalització.

    Args:
        soft_labels: matriu [vocab_size, vocab_size] de build_soft_labels()
    """

    def __init__(self, soft_labels: torch.Tensor):
        super().__init__()
        # register_buffer: guarda soft_labels com a part del model
        # (es desa al checkpoint i es mou a GPU amb .to(device)) però NO s'actualitza durant l'entrenament.
        self.register_buffer("soft_labels", soft_labels)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcula la loss semàntica per un batch.

        Args:
            logits:  prediccions del decoder (sense softmax) [N, vocab_size]
                     N = nombre total de tokens del batch (suma de totes les longituds)
            targets: índexs de les paraules correctes [N]

        Returns:
            loss: escalar — l'error mig del batch
        """
        # log_softmax és numèricament més estable que log(softmax(x))
        # Resultat: [N, vocab_size] amb valors ≤ 0
        log_probs = torch.log_softmax(logits, dim=1)  # [N, vocab_size]

        # Indexem la matriu de soft labels amb els targets:
        # per cada target[i], agafem la fila i de soft_labels.
        # Resultat: [N, vocab_size] — distribució target per a cada token del batch
        soft_tgts = self.soft_labels[targets]  # [N, vocab_size]

        # Cross-entropy suavitzada:
        # Per cada token: -suma_j( soft_tgt[j] * log_prob[j] )
        # .sum(dim=1) → [N]    .mean() → escalar
        return -(soft_tgts * log_probs).sum(dim=1).mean()