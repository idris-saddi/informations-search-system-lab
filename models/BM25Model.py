from collections import Counter, defaultdict
import math
from typing import Optional

from corpus import preprocess


class BM25Model:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lens = {doc_id: len(tokens) for doc_id, tokens in corpus.items()}
        self.avgdl = (sum(self.doc_lens.values()) / len(corpus)) if corpus else 0.0  # Longueur moyenne
        self.N = len(corpus)
        self.doc_freqs = self._compute_df()
        self._tf = {doc_id: Counter(tokens) for doc_id, tokens in corpus.items()}

    def _compute_df(self):
        df = Counter()
        for tokens in self.corpus.values():
            df.update(set(tokens))
        return df

    def _idf(self, term):
        # Formule IDF specifique BM25 [cite: 2380]
        n_qi = self.doc_freqs[term]
        return math.log((self.N - n_qi + 0.5) / (n_qi + 0.5) + 1)

    def search(self, query: str, top_k: Optional[int] = None):
        ranked = self._search_all(query)
        if top_k is not None:
            return ranked[:top_k]
        return ranked

    def _search_all(self, query: str):
        query_tokens = preprocess(query)
        scores = defaultdict(float)
        
        for doc_id, doc_tokens in self.corpus.items():
            doc_len = self.doc_lens.get(doc_id, len(doc_tokens))
            term_counts = self._tf.get(doc_id)
            if term_counts is None:
                term_counts = Counter(doc_tokens)
            
            for term in query_tokens:
                if term not in self.doc_freqs:
                    continue
                
                tf = term_counts[term]
                idf = self._idf(term)
                
                numerator = tf * (self.k1 + 1)
                if self.avgdl > 0:
                    denom_norm = (1 - self.b + self.b * (doc_len / self.avgdl))
                else:
                    denom_norm = 1.0
                denominator = tf + self.k1 * denom_norm
                scores[doc_id] += idf * (numerator / denominator)
                
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)