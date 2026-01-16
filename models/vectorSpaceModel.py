from collections import Counter
import math
from typing import Optional

import numpy as np

from corpus import preprocess


class VectorSpaceModel:
    def __init__(self, corpus, vocab):
        self.vocab = list(vocab)
        self.corpus = corpus
        self.N = len(corpus)
        self.idf = self._compute_idf()
        self._doc_vectors = {}
        self._doc_norms = {}
        for doc_id, tokens in self.corpus.items():
            vec = self._compute_tf_idf_vector(tokens)
            self._doc_vectors[doc_id] = vec
            self._doc_norms[doc_id] = float(np.linalg.norm(vec))
    
    def _compute_idf(self):
        idf_scores = {}
        for word in self.vocab:
            # df: nombre de documents contenant le terme [cite: 1953]
            df = sum(1 for tokens in self.corpus.values() if word in tokens)
            # idf = log(N / df) [cite: 1958]
            idf_scores[word] = math.log10(self.N / (df if df > 0 else 1))
        return idf_scores

    def _compute_tf_idf_vector(self, tokens):
        vec = np.zeros(len(self.vocab))
        token_counts = Counter(tokens)
        for i, word in enumerate(self.vocab):
            if word in token_counts:
                tf = token_counts[word]
                # Log normalization for TF: 1 + log10(tf) [cite: 1937]
                w_tf = 1 + math.log10(tf)
                vec[i] = w_tf * self.idf.get(word, 0)
        return vec

    def search(self, query: str, top_k: Optional[int] = None):
        query_tokens = preprocess(query)
        q_vec = self._compute_tf_idf_vector(query_tokens)
        q_norm = float(np.linalg.norm(q_vec))
        scores = {}
        
        for doc_id, d_vec in self._doc_vectors.items():
            d_norm = self._doc_norms.get(doc_id, 0.0)
            # Similarite Cosinus [cite: 1988]
            if q_norm > 0 and d_norm > 0:
                score = float(np.dot(q_vec, d_vec) / (q_norm * d_norm))
            else:
                score = 0.0
            scores[doc_id] = score
            
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            return ranked[:top_k]
        return ranked