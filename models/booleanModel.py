from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from corpus import STOPWORDS, preprocess


@dataclass(frozen=True)
class _BoolToken:
    kind: str  # TERM | AND | OR | NOT | LPAREN | RPAREN
    value: str = ""


def _tokenize_boolean_query(query: str) -> List[_BoolToken]:
    raw = query.strip()
    if not raw:
        return []

    # Tokenize while preserving parentheses and operators.
    parts: List[str] = []
    buf = ""
    for ch in raw:
        if ch in "()":
            if buf.strip():
                parts.extend(buf.strip().split())
            parts.append(ch)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.extend(buf.strip().split())

    tokens: List[_BoolToken] = []
    for p in parts:
        u = p.upper()
        if p == "(":
            tokens.append(_BoolToken("LPAREN"))
        elif p == ")":
            tokens.append(_BoolToken("RPAREN"))
        elif u in {"AND", "OR", "NOT"}:
            tokens.append(_BoolToken(u))
        else:
            # Normalize term with the same pipeline as the corpus.
            # We keep stopwords out to avoid useless constraints.
            norm_terms = preprocess(p, stopwords=STOPWORDS)
            if not norm_terms:
                continue
            # If a "term" yields multiple tokens (rare), join them with AND.
            first = True
            for t in norm_terms:
                if not first:
                    tokens.append(_BoolToken("AND"))
                tokens.append(_BoolToken("TERM", t))
                first = False
    return tokens


def _to_rpn(tokens: Sequence[_BoolToken]) -> List[_BoolToken]:
    # Shunting-yard: NOT > AND > OR
    prec = {"NOT": 3, "AND": 2, "OR": 1}
    out: List[_BoolToken] = []
    stack: List[_BoolToken] = []

    for tok in tokens:
        if tok.kind == "TERM":
            out.append(tok)
        elif tok.kind in {"AND", "OR", "NOT"}:
            while stack and stack[-1].kind in prec and prec[stack[-1].kind] >= prec[tok.kind]:
                out.append(stack.pop())
            stack.append(tok)
        elif tok.kind == "LPAREN":
            stack.append(tok)
        elif tok.kind == "RPAREN":
            while stack and stack[-1].kind != "LPAREN":
                out.append(stack.pop())
            if stack and stack[-1].kind == "LPAREN":
                stack.pop()
    while stack:
        out.append(stack.pop())
    return out


def _eval_rpn(rpn: Sequence[_BoolToken], doc_terms: Iterable[str]) -> bool:
    doc_set = set(doc_terms)
    stack: List[bool] = []
    for tok in rpn:
        if tok.kind == "TERM":
            stack.append(tok.value in doc_set)
        elif tok.kind == "NOT":
            a = stack.pop() if stack else False
            stack.append(not a)
        elif tok.kind == "AND":
            b = stack.pop() if stack else False
            a = stack.pop() if stack else False
            stack.append(a and b)
        elif tok.kind == "OR":
            b = stack.pop() if stack else False
            a = stack.pop() if stack else False
            stack.append(a or b)
    return bool(stack[-1]) if stack else False


class BooleanModel:
    """Modele booleen sur corpus tokenise.

    - Supporte AND/OR/NOT et parentheses.
    - Si aucun operateur n'est fourni, on applique une conjonction (AND) implicite.
    """

    def __init__(self, corpus: Dict[str, List[str]]):
        self.corpus = corpus

    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        tokens = _tokenize_boolean_query(query)
        if not tokens:
            return []

        # AND implicite entre deux TERM consecutifs.
        expanded: List[_BoolToken] = []
        for tok in tokens:
            if expanded and expanded[-1].kind == "TERM" and tok.kind == "TERM":
                expanded.append(_BoolToken("AND"))
            expanded.append(tok)

        rpn = _to_rpn(expanded)
        results: List[Tuple[str, float]] = []
        for doc_id, doc_terms in self.corpus.items():
            if _eval_rpn(rpn, doc_terms):
                results.append((doc_id, 1.0))

        # Deterministic order: by doc_id (scores identical)
        results.sort(key=lambda x: x[0])
        if top_k is not None:
            return results[:top_k]
        return results