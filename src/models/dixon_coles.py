# src/models/dixon_coles.py
import numpy as np
from math import factorial

def _poisson_pmf(k, lam):
    return np.exp(-lam) * (lam ** k) / factorial(k)

def score_matrix_dc(lambda_home, lambda_away, max_goals=8, rho=0.12):
    """
    Matriz de prob de marcadores con corrección Dixon–Coles “light”.
    rho ~ [0.05, 0.2]; 0.12 suele ir bien como default genérico.
    """
    size = max_goals + 1
    M = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            base = _poisson_pmf(i, lambda_home) * _poisson_pmf(j, lambda_away)
            # Ajustes DC solo en las celdas más sensibles (0-0,1-0,0-1,1-1)
            if i == 0 and j == 0:
                adj = 1 - (lambda_home * lambda_away * rho)
            elif i == 0 and j == 1:
                adj = 1 + (lambda_home * rho)
            elif i == 1 and j == 0:
                adj = 1 + (lambda_away * rho)
            elif i == 1 and j == 1:
                adj = 1 - rho
            else:
                adj = 1.0
            M[i, j] = base * max(0.0, adj)  # proteger de negativos
    M /= M.sum()
    return M

def probs_from_matrix(M):
    p_home = np.tril(M, -1).sum()
    p_draw = np.trace(M)
    p_away = np.triu(M, 1).sum()
    return float(p_home), float(p_draw), float(p_away)

def prob_over(M, line=2.5):
    p = 0.0
    size = M.shape[0]
    for i in range(size):
        for j in range(size):
            if i + j > line:
                p += M[i, j]
    return float(p)
