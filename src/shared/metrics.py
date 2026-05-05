"""Shared metrics for image captioning evaluation."""
from __future__ import annotations
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

def lcs(a: list[str], b: list[str]) -> int:
    """Calcula la longitud de la seqüència comuna més llarga (Longest Common Subsequence)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def rouge_l_score(references: list[list[str]], hypothesis: list[str]) -> float:
    """Calcula el score ROUGE-L (F1-score basat en LCS)."""
    if not hypothesis:
        return 0.0
    lcs_lens = [lcs(ref, hypothesis) for ref in references]
    best_lcs = max(lcs_lens)
    prec = best_lcs / len(hypothesis)
    rec = best_lcs / np.mean([len(r) for r in references])
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)
