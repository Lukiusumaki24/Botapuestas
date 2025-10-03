import argparse, pandas as pd, yaml, numpy as np
from elo import build_elo
from poisson import match_score_matrix, probs_1x2_from_matrix, prob_over_from_matrix
from odds import implied_prob_from_decimal, remove_overround, kelly_fraction

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

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hist', required=True)
    p.add_argument('--upcoming', required=True)
    p.add_argument('--config', default='config/leagues.yaml')
    p.add_argument('--out', default='data/picks_all.csv')
    p.add_argument('--value_threshold', type=float, default=0.02)
    p.add_argument('--kelly_fraction', type=float, default=0.25)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    patterns = cfg.get('league_patterns', [])

    hist = pd.read_csv(args.hist, parse_dates=['date'])
    upcoming = pd.read_csv(args.upcoming, parse_dates=['date'])
    mask = False
    for pat in patterns:
        mask = mask | upcoming['league'].str.contains(pat, case=False, na=False)
    up = upcoming[mask].copy()

    elo = build_elo(hist); form_tbl = recent_form(hist, N=6)
    picks = []
    for _, row in up.iterrows():
        lam_h, lam_a = estimate_goal_rates(row, elo, form_tbl)
        M = match_score_matrix(lam_h, lam_a, max_goals=8)
        pH, pD, pA = probs_1x2_from_matrix(M)
        pOver25 = prob_over_from_matrix(M, 2.5); pUnder25 = 1 - pOver25

        p_h_imp, p_d_imp, p_a_imp = implied_prob_from_decimal([row['home_odds'], row['draw_odds'], row['away_odds']])
        p_h_imp, p_d_imp, p_a_imp = remove_overround(p_h_imp, p_d_imp, p_a_imp)

        umbral = args.value_threshold
        values = []
        if pH > p_h_imp + umbral: values.append(('Home', pH, row['home_odds']))
        if pD > p_d_imp + umbral: values.append(('Draw', pD, row['draw_odds']))
        if pA > p_a_imp + umbral: values.append(('Away', pA, row['away_odds']))

        if 'ou25_over_odds' in up.columns and not pd.isna(row.get('ou25_over_odds', float('nan'))):
            p_over_imp = 1/row['ou25_over_odds']; p_under_imp = 1/row['ou25_under_odds']
            s = p_over_imp + p_under_imp
            p_over_imp, p_under_imp = p_over_imp/s, p_under_imp/s
            if pOver25 > p_over_imp + umbral: values.append(('Over 2.5', pOver25, row['ou25_over_odds']))
            if pUnder25 > p_under_imp + umbral: values.append(('Under 2.5', pUnder25, row['ou25_under_odds']))

        for market, p_mod, odds in values:
            f = kelly_fraction(p_mod, odds) * args.kelly_fraction
            picks.append({
                'date': row['date'].date().isoformat(),
                'league': row.get('league',''),
                'match': f"{row['home']} vs {row['away']}",
                'market': market,
                'p_model': round(p_mod,4),
                'odds': odds,
                'kelly_frac': round(f,4)
            })

    pd.DataFrame(picks).sort_values(['date','league','match']).to_csv(args.out, index=False)
    print(f"Guardado en {args.out}")
print('run multileague placeholder')
