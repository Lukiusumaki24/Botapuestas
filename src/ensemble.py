import numpy as np
from poisson import match_score_matrix as poisson_matrix, probs_1x2_from_matrix as poisson_probs, prob_over_from_matrix as poisson_over
from models.dixon_coles import score_matrix_dc, probs_from_matrix as dc_probs, prob_over as dc_over

def softmax3(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

def bradley_terry_probs(elo_home, elo_away, home_adv=60.0, k_draw=0.22):
    gap = (elo_home + home_adv) - elo_away
    z_home = gap / 120.0
    z_away = -gap / 120.0
    z_draw = 0.0
    p = softmax3(np.array([z_home, z_draw, z_away]))
    p[1] = p[1] + k_draw * np.exp(-abs(gap)/300.0)
    p /= p.sum()
    return float(p[0]), float(p[1]), float(p[2])

def ensemble_predict(lh, la, elo_h, elo_a, weights=(0.4, 0.35, 0.25), rho=0.12):
    Mp = poisson_matrix(lh, la, max_goals=8)
    pH_p, pD_p, pA_p = poisson_probs(Mp)
    over_p = poisson_over(Mp, 2.5)

    Md = score_matrix_dc(lh, la, max_goals=8, rho=rho)
    pH_d, pD_d, pA_d = dc_probs(Md)
    over_d = dc_over(Md, 2.5)

    pH_e, pD_e, pA_e = bradley_terry_probs(elo_h, elo_a)

    w1, w2, w3 = weights
    pH = w1*pH_p + w2*pH_d + w3*pH_e
    pD = w1*pD_p + w2*pD_d + w3*pD_e
    pA = w1*pA_p + w2*pA_d + w3*pA_e
    s = pH + pD + pA
    pH, pD, pA = pH/s, pD/s, pA/s
    pOver = 0.5*over_p + 0.5*over_d
    return {
        "pH": float(pH), "pD": float(pD), "pA": float(pA),
        "pOver25": float(pOver),
        "components": {
            "poisson": {"pH": pH_p, "pD": pD_p, "pA": pA_p, "pOver25": over_p},
            "dixon_coles": {"pH": pH_d, "pD": pD_d, "pA": pA_d, "pOver25": over_d},
            "elo_bt": {"pH": pH_e, "pD": pD_e, "pA": pA_e}
        }
    }
