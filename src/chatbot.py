import argparse, re, pandas as pd, numpy as np
from datetime import datetime, timedelta
import pytz
from elo import build_elo
from poisson import match_score_matrix, probs_1x2_from_matrix, prob_over_from_matrix
from odds import implied_prob_from_decimal, remove_overround, kelly_fraction

TZ = pytz.timezone("America/Bogota")

def parse_date_from_query(q):
    q = q.lower()
    today = datetime.now(TZ).date()
    if "hoy" in q: return today
    if "mañana" in q: return today + timedelta(days=1)
    m = re.search(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", q)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        y = int(y) if y else today.year
        if y < 100: y += 2000
        return datetime(y, mo, d, tzinfo=TZ).date()
    return today

def recent_form(df_hist, N=6):
    form = {}
    teams = set(df_hist['home']).union(set(df_hist['away']))
    for t in teams:
        rows_h = df_hist[df_hist['home']==t].sort_values('date').tail(N)
        rows_a = df_hist[df_hist['away']==t].sort_values('date').tail(N)
        gf = rows_h['home_goals'].sum() + rows_a['away_goals'].sum()
        ga = rows_h['away_goals'].sum() + rows_a['home_goals'].sum()
        games = len(rows_h)+len(rows_a)
        val = (gf - ga)/max(1, games)
        form[(t,'F')] = val
    return form

def estimate_goal_rates(row, rating, form_tbl, base_lambda=1.35, beta_elo=0.0035, beta_form=0.08, home_bias=0.15):
    Rh = rating.get(row['home'],1500)
    Ra = rating.get(row['away'],1500)
    elo_diff = (Rh - Ra)/100.0
    form_h = form_tbl.get((row['home'],'F'),0.0)
    form_a = form_tbl.get((row['away'],'F'),0.0)
    lam_h = max(0.05, base_lambda * np.exp(beta_elo*elo_diff + beta_form*form_h + home_bias))
    lam_a = max(0.05, base_lambda * np.exp(-beta_elo*elo_diff + beta_form*form_a))
    return lam_h, lam_a

def double_chance_probs(pH, pD, pA):
    return {"1X": pH + pD, "12": pH + pA, "X2": pD + pA}

def main(args):
    target_date = parse_date_from_query(args.question)
    hist = pd.read_csv(args.hist, parse_dates=['date'])
    upcoming = pd.read_csv(args.upcoming, parse_dates=['date'])

    up = upcoming[upcoming['date'].dt.date == target_date].copy()
    if len(up)==0:
        print(f"No hay partidos para {target_date.isoformat()}.")
        return

    elo = build_elo(hist)
    form_tbl = recent_form(hist, N=6)

    for _, row in up.iterrows():
        lam_h, lam_a = estimate_goal_rates(row, elo, form_tbl)
        M = match_score_matrix(lam_h, lam_a, max_goals=8)
        pH, pD, pA = probs_1x2_from_matrix(M)
        pOver25 = prob_over_from_matrix(M, 2.5)
        dc = double_chance_probs(pH,pD,pA)
        best_dc = max(dc.items(), key=lambda x: x[1])
        print(f"- {row.get('league','')}: {row['home']} vs {row['away']}")
        print(f"  ▸ 1X2: Local {pH:.3f}, Empate {pD:.3f}, Visita {pA:.3f}")
        print(f"  ▸ Doble Oportunidad: {best_dc[0]} (p≈{best_dc[1]:.3f})")
        print(f"  ▸ Goles: Over 2.5 p≈{pOver25:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", help="Ej: 'qué apuestas hay para hoy?'")
    p.add_argument("--hist", default="data/matches_hist.csv")
    p.add_argument("--upcoming", default="data/upcoming.csv")
    args = p.parse_args()
    main(args)
