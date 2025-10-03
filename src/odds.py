import numpy as np

def implied_prob_from_decimal(odds):
    return 1.0 / np.array(odds, dtype=float)

def remove_overround(p_home, p_draw, p_away):
    s = p_home + p_draw + p_away
    return p_home/s, p_draw/s, p_away/s

def kelly_fraction(p, odds_decimal):
    b = odds_decimal - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b*p - (1-p)) / b)
