"""Transparent binary relevance metrics for top-K ranking evaluation."""
from __future__ import annotations
import math
from itertools import combinations


def ranking_metrics(recommended: list[int], relevant: set[int], k: int) -> dict[str,float]:
    """Precision denominator is K; ideal DCG uses min(K, |relevant|)."""
    if k < 1 or not relevant:
        raise ValueError('k must be positive and relevant items must be nonempty')
    if len(recommended) > k or len(set(recommended)) != len(recommended):
        raise ValueError('Recommendations exceed k or contain duplicates')
    hits=[int(x in relevant) for x in recommended]
    dcg=sum(hit/math.log2(i+2) for i,hit in enumerate(hits))
    idcg=sum(1/math.log2(i+2) for i in range(min(k,len(relevant))))
    return {'precision':sum(hits)/k,'recall':sum(hits)/len(relevant),'ndcg':dcg/idcg}


def genre_diversity(recommended: list[int], genres: dict[int,frozenset[str]]) -> float:
    """Mean pairwise Jaccard genre distance. Empty/unknown genres count as distance 0."""
    if len(recommended)<2:
        return 0.0
    total=0.0
    for a,b in combinations(recommended,2):
        if a not in genres or b not in genres:
            raise ValueError('Recommendation item missing from genre map')
        ga,gb=genres[a],genres[b]
        union=ga|gb
        total+=1-len(ga&gb)/len(union) if union else 0.0
    return total/math.comb(len(recommended),2)
