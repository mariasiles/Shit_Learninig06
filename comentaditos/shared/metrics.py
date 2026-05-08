"""
metrics.py
==========
Mètriques d'avaluació per a image captioning.

Per què necessitem mètriques especialitzades?
    Avaluar una caption generada no és trivial. Una caption pot ser bona però
    usar paraules sinònimes a les de la referència, o reordenar-les.
    Cal una mesura que vagi més enllà de la coincidència exacta de caràcters.

Mètriques implementades:
    LCS / ROUGE-L:
        Mida de la seqüència de paraules comunes més llarga entre la caption
        generada i la de referència, en el mateix ordre però no necessàriament
        consecutives. Més flexible que BLEU (no exigeix blocs contigus).

    BLEU i METEOR:
        Importades de NLTK. BLEU mesura coincidències de n-grams (blocs de n
        paraules consecutives). METEOR també considera sinònims i formes derivades.

Diferència respecte al BLEU original del train.py:
    El train.py ja calcula BLEU-1 i BLEU-4 directament.
    Aquest mòdul hi afegeix ROUGE-L i encapsula totes les mètriques en una
    funció full_eval_report() que retorna un diccionari de resultats.
"""

from __future__ import annotations
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score


# ─────────────────────────────────────────────────────────────────────────────
# LCS — LONGEST COMMON SUBSEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

def lcs(a: list[str], b: list[str]) -> int:
    """
    Calcula la longitud de la Seqüència Comuna Més Llarga (LCS) entre a i b.

    Definició de LCS:
        Donades dues seqüències, la LCS és la seqüència més llarga de elements
        que apareix en AMBDUES en el MATEIX ORDRE, però no necessàriament
        en posicions consecutives.

    Exemple:
        a = ["the", "dog", "runs", "fast"]
        b = ["a",   "dog", "is",   "fast"]
        LCS = ["dog", "fast"] → longitud 2

    Com funciona (programació dinàmica)?
        Construïm una taula dp on dp[i][j] = longitud de la LCS de a[:i] i b[:j].
        Règles de transició:
          - Si a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1] + 1  (el caràcter coincideix, sumem 1)
          - Altrament:          dp[i][j] = max(dp[i-1][j], dp[i][j-1])  (el millor sense incloure'l)

        La resposta és dp[n][m] (n=longitud de a, m=longitud de b).
        Cost: O(n·m) temps i memòria.

    Args:
        a, b: llistes de strings (tokens/paraules)

    Returns:
        longitud de la LCS (enter ≥ 0)
    """
    n, m = len(a), len(b)

    # Taula dp de (n+1) × (m+1) inicialitzada a zero.
    # La fila 0 i la columna 0 representen seqüències buides → LCS = 0.
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n):         # itera sobre cada token de a
        for j in range(m):     # itera sobre cada token de b
            if a[i] == b[j]:
                # Els tokens coincideixen: la LCS fins aquí és la LCS de a[:i] i b[:j] més 1 (el token actual)
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                # No coincideixen: prenem el millor dels dos casos:
                # ignorar el token actual de a, o ignorar el de b
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    return dp[n][m]  # longitud de la LCS completa


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE-L
# ─────────────────────────────────────────────────────────────────────────────

def rouge_l_score(references: list[list[str]], hypothesis: list[str]) -> float:
    """
    Calcula el score ROUGE-L (F1-score basat en LCS) per a una imatge.

    ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation — Longest):
        Mesura la qualitat d'una caption comparant-la amb les captions de
        referència humanes. No exigeix coincidència exacta de blocs contigus
        (com BLEU), sinó coincidència de la seqüència més llarga en ordre.

    Per a múltiples referències (Flickr8k té 5 per imatge):
        Calculem ROUGE-L per a CADA referència i retornem el màxim.
        Lògica: si la hipòtesi coincideix bé amb ALGUNA referència, és bona.

    Fórmula F1:
        precision = lcs_length / len(hypothesis)   (quant de la hipòtesi és correcte)
        recall    = lcs_length / len(reference)    (quant de la referència es cobreix)
        F1        = 2 · precision · recall / (precision + recall)

    Args:
        references: llista de captions de referència, cada una tokenitzada
                    Ex: [["a","dog","runs"], ["the","dog","is","running"]]
        hypothesis: caption generada pel model, tokenitzada
                    Ex: ["a","dog","is","running"]

    Returns:
        ROUGE-L F1-score on 0.0 = cap coincidència, 1.0 = coincidència perfecta
    """
    # Si la hipòtesi és buida ([]), no hi ha res a comparar.
    # Retornem directament score 0.
    if not hypothesis:
        return 0.0
    
    # Calculem la longitud de la LCS entre la hipòtesi i cadascuna de les referències.
    lcs_lens = [lcs(ref, hypothesis) for ref in references]

    # Ens quedem amb la millor coincidència entre totes les referències.
    # Si tenim diverses captions correctes per una mateixa imatge,
    # només utilitzem la que millor encaixa amb la generada.
    best_lcs = max(lcs_lens)
    
    prec = best_lcs / len(hypothesis)   # precision: quina part de la hipòtesi és correcta
    rec = best_lcs / np.mean([len(r) for r in references])  # recall: quina part de les referències hem recuperat (usant longitud mitjana)

    # Evitem divisió per zero.
    # Si precision = 0 i recall = 0,
    # el denominador de la F1 seria 0.
    if prec + rec == 0:
        return 0.0
    
    return 2 * prec * rec / (prec + rec)   # F1-score: mitja harmònica de precision i recall