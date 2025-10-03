import numpy as np

def update_elo(r_home, r_away, goals_h, goals_a, K=20, home_adv=60):
    Eh = 1/(1+10**(((r_away)-(r_home+home_adv))/400))
    Ea = 1-Eh
    if goals_h > goals_a: Sh, Sa = 1, 0
    elif goals_h == goals_a: Sh, Sa = 0.5, 0.5
    else: Sh, Sa = 0, 1
    mov = max(1, abs(goals_h-goals_a))
    mult = np.log(mov+1)*2.2/((abs((r_home+home_adv)-r_away)*0.001)+2.2)
    r_home_new = r_home + K*mult*(Sh - Eh)
    r_away_new = r_away + K*mult*(Sa - Ea)
    return r_home_new, r_away_new

def build_elo(df):
    rating = {}
    for _, row in df.sort_values('date').iterrows():
        h, a = row['home'], row['away']
        r_h = rating.get(h, 1500.0)
        r_a = rating.get(a, 1500.0)
        r_h2, r_a2 = update_elo(r_h, r_a, row['home_goals'], row['away_goals'])
        rating[h], rating[a] = r_h2, r_a2
    return rating
