# app/embedder.py
import re, math
from collections import Counter

_token_pattern = re.compile(r"\b\w+\b")

def tokenize(text: str) -> list[str]:
    return _token_pattern.findall(text.lower())

def build_tfidf(cats_by_market: dict[str, list[dict]]):
    """
    각 마켓의 카테고리 text를 토큰화하여 TF-IDF 벡터를 계산하고,
    idf와 tfidf 벡터 목록을 리턴합니다.
    """
    idf_by_market = {}
    tfidf_by_market = {}
    for market, cats in cats_by_market.items():
        docs = [tokenize(c["text"]) for c in cats]
        N = len(docs)
        # 문서 빈도 계산
        df = Counter()
        for doc in docs:
            for term in set(doc):
                df[term] += 1
        # IDF 계산
        idf = {t: math.log(N/df[t]) for t in df}
        idf_by_market[market] = idf
        # TF-IDF 벡터 계산
        tfidf_vecs = []
        for doc in docs:
            tf = Counter(doc)
            total = len(doc) or 1
            vec = {t: (tf[t]/total) * idf.get(t, 0.0) for t in tf}
            tfidf_vecs.append(vec)
        tfidf_by_market[market] = tfidf_vecs
    return idf_by_market, tfidf_by_market

def embed_query(text: str, idf: dict[str, float]) -> dict[str, float]:
    """
    쿼리 텍스트를 받아 TF-IDF 벡터(dict)로 변환합니다.
    """
    tokens = tokenize(text)
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {t: (tf[t]/total) * idf.get(t, 0.0) for t in tf}

def cosine_similarity(u: dict[str, float], v: dict[str, float]) -> float:
    """
    두 TF-IDF 벡터 간 코사인 유사도를 계산합니다.
    """
    num = sum(u.get(k,0.0) * v.get(k,0.0) for k in u)
    den = math.sqrt(sum(val*val for val in u.values())) \
        * math.sqrt(sum(val*val for val in v.values()))
    return num/den if den else 0.0
