"""Semantic-aware loss functions for image captioning."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_soft_labels(embed_weights: torch.Tensor, temperature: float = 10.0) -> torch.Tensor:
    """Construeix una matriu de soft labels semàntics a partir dels pesos d'embedding.

    Per a cada paraula i, soft_labels[i] és una distribució de probabilitat sobre
    el vocabulari proporcional a la similitud cosinus entre i i totes les altres paraules.

    Exemple: si el target és "mountain", paraules com "hill" o "peak" reben
    probabilitat positiva en lloc de zero (com fa CrossEntropy estàndard).

    Args:
        embed_weights: matriu d'embeddings [vocab_size, embed_dim] (GloVe o aleatoris)
        temperature:   controla la concentració. Més alt → més concentrat al target.
                       Recomanat: 10.0 amb GloVe.

    Returns:
        soft_labels [vocab_size, vocab_size] — cada fila és una distribució de prob.
    """
    norms = embed_weights.norm(dim=1, keepdim=True).clamp(min=1e-8)  # norma L2 de cada vector d'embedding, clampejada per evitar divisió per zero
    normalized = embed_weights / norms                    # normalitza cada vector → vectors unitaris per calcular similitud cosinus
    sim = normalized @ normalized.T                       # producte matricial → [vocab_size, vocab_size] on sim[i,j] = cosinus entre paraula i i paraula j
    return torch.softmax(sim * temperature, dim=1)        # aplica softmax amb temperatura → cada fila és una distribució de prob sobre el vocabulari


class SemanticCrossEntropyLoss(nn.Module):
    """CrossEntropy amb soft labels semàntics basats en similitud d'embeddings GloVe.

    Penalitza menys les prediccions de paraules semànticament similars al target.
    Quan soft_labels és one-hot (temp→∞), és equivalent a CrossEntropy estàndard.
    """

    def __init__(self, soft_labels: torch.Tensor):
        super().__init__()
        self.register_buffer("soft_labels", soft_labels)  # registra com a buffer → es guarda al checkpoint però no és un paràmetre entrenable

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits:  [N, vocab_size] → puntuacions crues del model per a cada paraula del vocabulari
        # targets: [N]  → índexs de les paraules target (la paraula correcta de cada pas)
        log_probs = torch.log_softmax(logits, dim=1)      # converteix logits a log-probabilitats → [N, vocab_size]
        soft_tgts = self.soft_labels[targets]              # agafa la fila de soft labels corresponent a cada target → [N, vocab_size]
        return -(soft_tgts * log_probs).sum(dim=1).mean()  # cross-entropy generalitzada: -sum(soft_target * log_prob) → escalar
