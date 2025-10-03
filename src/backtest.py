import argparse, pandas as pd, numpy as np

def kelly_fraction(p, odds_decimal):
    b = odds_decimal - 1.0
    if b <= 0: 
        return 0.0
    return max(0.0, (b*p - (1-p)) / b)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hist', required=True)
    p.add_argument('--odds', required=True)
    p.add_argument('--out', default='data/backtest_report.csv')
    args = p.parse_args()

    hist = pd.read_csv(args.hist, parse_dates=['date'])
    odds = pd.read_csv(args.odds, parse_dates=['date'])
    df = hist.merge(odds, on=['date','home','away'], how='inner')

    def result(row):
        if row['home_goals']>row['away_goals']: return 'Home'
        if row['home_goals']<row['away_goals']: return 'Away'
        return 'Draw'
    df['result'] = df.apply(result, axis=1)

    df['p_home'] = 1/df['home_odds']; df['p_draw'] = 1/df['draw_odds']; df['p_away'] = 1/df['away_odds']
    s = df['p_home']+df['p_draw']+df['p_away']; df['p_home']/=s; df['p_draw']/=s; df['p_away']/=s

    th = 0.02
    bank = 1000.0
    bank_series = [bank]

    for i, r in df.iterrows():
        bets = []
        for mk, p_m, imp, odds_d in [('Home', r['p_home'], 1/r['home_odds']/s.loc[i], r['home_odds']),
                                     ('Draw', r['p_draw'], 1/r['draw_odds']/s.loc[i], r['draw_odds']),
                                     ('Away', r['p_away'], 1/r['away_odds']/s.loc[i], r['away_odds'])]:
            if p_m > imp + th:
                f = 0.25*kelly_fraction(p_m, odds_d)
                stake = bank * f
                bets.append((mk, stake, odds_d))
        pnl = 0.0
        for mk, stake, odds_d in bets:
            pnl += stake*(odds_d-1) if mk==r['result'] else -stake
        bank += pnl
        bank_series.append(bank)

    out = pd.DataFrame({'step': range(len(bank_series)), 'bank': bank_series})
    summary = {
        'final_bank': bank_series[-1],
        'roi_%': (bank_series[-1]-bank_series[0])/bank_series[0]*100,
        'max_drawdown_%': (1 - (pd.Series(bank_series)/pd.Series(bank_series).cummax()).min())*100
    }
    pd.DataFrame([summary]).to_csv(args.out, index=False)
    print('Resumen:', summary)
print('backtest placeholder')
