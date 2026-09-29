"""Self-BLEU exactly as utils.eval.self_bleu computes it, on saved texts (for comparing nltk versions)."""
import json
import sys

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

texts = [c.split() for c in json.load(open(sys.argv[1]))]
n_grams = 4
weights = [1.0 / n_grams] * n_grams
scores = []
for i in range(len(texts)):
    refs = [list(texts[j]) for j in range(len(texts)) if j != i]
    scores.append(sentence_bleu(refs, list(texts[i]), weights=weights, smoothing_function=SmoothingFunction().method3))
print(f"nltk {nltk.__version__}: Self-BLEU-4 = {sum(scores) / len(scores):.4f}")
