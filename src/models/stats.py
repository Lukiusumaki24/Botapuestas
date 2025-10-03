import numpy as np
from math import factorial

def _pmf(k, lam): 
    return np.exp(-lam)*lam**k/factorial(k)

def poisson_over_prob(lam, line=9.5, max_k=35):
    lam = float(lam) if lam is not None else 0.0
    if lam <= 0: 
        return 0.0
    p = 0.0
    for k in range(0, max_k+1):
        if k > line: p += _pmf(k, lam)
    return float(min(1.0, max(0.0, p)))

def poisson_exact_prob(lam, k, max_k=35):
    lam = float(lam) if lam is not None else 0.0
    if lam <= 0: 
        return 0.0
    return float(_pmf(k, lam))

def combine_home_away_rate(lh, la):
    return float(max(0.01, (0 if lh is None else lh) + (0 if la is None else la)))
