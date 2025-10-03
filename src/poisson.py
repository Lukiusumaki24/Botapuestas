import numpy as np
from math import factorial

def poisson_pmf(k, lam):
    return np.exp(-lam)*lam**k/factorial(k)

def match_score_matrix(lh, la, max_goals=10):
    m = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            m[i,j] = poisson_pmf(i, lh)*poisson_pmf(j, la)
    m /= m.sum()
    return m

def probs_1x2_from_matrix(M):
    p_home = np.tril(M, -1).sum()
    p_draw = np.trace(M)
    p_away = np.triu(M, 1).sum()
    return p_home, p_draw, p_away

def prob_over_from_matrix(M, line=2.5):
    p = 0.0
    size = M.shape[0]
    for i in range(size):
        for j in range(size):
            if i+j > line:
                p += M[i,j]
    return p
