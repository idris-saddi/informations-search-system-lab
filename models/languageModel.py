from collections import Counter
import math
from typing import Dict, Optional

from corpus import preprocess


class LanguageModelJM:
    def __init__(self, corpus, lam=0.5): # Lambda parameter
        self.corpus = corpus
        self.lam = lam
        self.doc_lens = {doc_id: len(tokens) for doc_id, tokens in corpus.items()}
        self.doc_counts = {doc_id: Counter(tokens) for doc_id, tokens in corpus.items()}
        # Modele de la collection (Mc) 
        all_tokens = [t for tokens in corpus.values() for t in tokens]
        self.collection_len = len(all_tokens)
        collection_counts = Counter(all_tokens)
        if self.collection_len > 0:
            self.collection_probs: Dict[str, float] = {
                term: count / self.collection_len for term, count in collection_counts.items()
            }
        else:
            self.collection_probs = {}

    def search(self, query: str, top_k: Optional[int] = None):
        query_tokens = preprocess(query)
        scores = {}
        
        for doc_id, doc_tokens in self.corpus.items():
            doc_len = self.doc_lens.get(doc_id, len(doc_tokens))
            doc_probs = self.doc_counts.get(doc_id)
            if doc_probs is None:
                doc_probs = Counter(doc_tokens)
            log_score = 0.0
            
            for term in query_tokens:
                # P(qi | Md)
                p_term_doc = doc_probs[term] / doc_len if doc_len > 0 else 0
                # P(qi | Mc)
                p_term_col = self.collection_probs.get(term, 0.0)
                
                # Lissage Jelinek-Mercer [cite: 1663]
                smoothed_prob = ((1 - self.lam) * p_term_doc) + (self.lam * p_term_col)

                # Work in log-space to avoid underflow when multiplying tiny probabilities.
                smoothed_prob = max(smoothed_prob, 1e-12)
                log_score += math.log(smoothed_prob)
            
            scores[doc_id] = log_score
            
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            return ranked[:top_k]
        return ranked