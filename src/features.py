import pandas as pd
import numpy as np

def rolling_team_stats(hist: pd.DataFrame, team: str, n: int = 6):
    h = hist[hist['home']==team].sort_values('date').tail(n)
    a = hist[hist['away']==team].sort_values('date').tail(n)
    def safe_mean(s): 
        s = pd.to_numeric(s, errors='coerce')
        return float(s.dropna().mean()) if len(s.dropna()) else None
    stats = {}
    gf = (h['home_goals'].sum() + a['away_goals'].sum())
    ga = (h['away_goals'].sum() + a['home_goals'].sum())
    games = len(h)+len(a)
    stats['gdpg'] = (gf - ga) / max(1, games)
    stats['gfpg'] = gf / max(1, games)
    stats['gapg'] = ga / max(1, games)
    stats['corners_for_pg'] = np.nan
    ch = safe_mean(h['home_corners']); ca = safe_mean(a['away_corners'])
    if ch is not None and ca is not None:
        stats['corners_for_pg'] = (ch + ca) / 2.0
    stats['shots_for_pg'] = np.nan
    stats['shots_on_pg'] = np.nan
    sh = safe_mean(h['home_shots']); sa = safe_mean(a['away_shots'])
    so_h = safe_mean(h['home_shots_on']); so_a = safe_mean(a['away_shots_on'])
    if sh is not None and sa is not None:
        stats['shots_for_pg'] = (sh + sa)/2.0
    if so_h is not None and so_a is not None:
        stats['shots_on_pg'] = (so_h + so_a)/2.0
    stats['cards_for_pg'] = np.nan
    cd_h = safe_mean(h['home_cards']); cd_a = safe_mean(a['away_cards'])
    if cd_h is not None and cd_a is not None:
        stats['cards_for_pg'] = (cd_h + cd_a)/2.0
    stats['poss_pg'] = np.nan
    ph = safe_mean(h['home_possession']); pa = safe_mean(a['away_possession'])
    if ph is not None and pa is not None:
        stats['poss_pg'] = (ph + pa)/2.0
    stats['fouls_pg'] = np.nan
    fh = safe_mean(h['home_fouls']); fa = safe_mean(a['away_fouls'])
    if fh is not None and fa is not None:
        stats['fouls_pg'] = (fh + fa)/2.0
    return stats
