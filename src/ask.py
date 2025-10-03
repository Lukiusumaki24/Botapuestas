import yaml, pandas as pd, numpy as np, requests
from datetime import datetime, timedelta
import pytz
from elo import build_elo
from poisson import match_score_matrix, probs_1x2_from_matrix, prob_over_from_matrix

TZ = pytz.timezone("America/Bogota")

def load_config(path="config/api_keys.yaml"):
    try:
        return yaml.safe_load(open(path))
    except Exception:
        return {}

def api_get(url, headers=None, params=None, timeout=25):
    r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_today_from_api(cfg):
    api = cfg.get("football_data_org", {})
    token = api.get("api_key")
    comps = api.get("competitions", ["PL","PD","SA","BL1","FL1","DED","PPL"])
    if not token:
        return None, None, None

    import math
    from datetime import timedelta
    API_BASE = "https://api.football-data.org/v4"
    headers = {"X-Auth-Token": token}

    today = datetime.now(TZ).date()
    start = today - timedelta(days=180)

    hist_rows, up_rows = [], []
    window = timedelta(days=9)  # <= 10 días por ventana

    for comp in comps:
        # 1) Histórico en ventanas de 10 días
        d = start
        while d <= today:
            to = min(d + window, today)
            try:
                m = api_get(
                    f"{API_BASE}/competitions/{comp}/matches",
                    headers,
                    {"dateFrom": d.isoformat(), "dateTo": to.isoformat()},
                )
                for match in m.get("matches", []):
                    if match.get("status") != "FINISHED":
                        continue
                    sc = (match.get("score", {}) or {}).get("fullTime", {})
                    hist_rows.append({
                        "date": (match.get("utcDate","")[:10]),
                        "league": m.get("competition",{}).get("name", comp),
                        "home": match.get("homeTeam",{}).get("name"),
                        "away": match.get("awayTeam",{}).get("name"),
                        "home_goals": sc.get("home", None),
                        "away_goals": sc.get("away", None),
                        "home_corners": None, "away_corners": None,
                    })
            except Exception as e:
                # Si una ventana falla, seguimos con la siguiente para no romper todo
                pass
            d = to + timedelta(days=1)

        # 2) Partidos de HOY (TIMED/SCHEDULED)
        try:
            up = api_get(
                f"{API_BASE}/competitions/{comp}/matches",
                headers,
                {"dateFrom": today.isoformat(), "dateTo": today.isoformat()},
            )
            for match in up.get("matches", []):
                if match.get("status") not in ("TIMED", "SCHEDULED"):
                    continue
                up_rows.append({
                    "date": (match.get("utcDate","")[:10]),
                    "league": up.get("competition",{}).get("name", comp),
                    "home": match.get("homeTeam",{}).get("name"),
                    "away": match.get("awayTeam",{}).get("name"),
                    "home_odds": None, "draw_odds": None, "away_odds": None,
                    "ou25_over_odds": None, "ou25_under_odds": None,
                })
        except Exception:
            pass

    return pd.DataFrame(hist_rows), pd.DataFrame(up_rows), pd.DataFrame()


def recent_form(df_hist, team, N=6):
    h = df_hist[df_hist['home']==team].sort_values('date').tail(N)
    a = df_hist[df_hist['away']==team].sort_values('date').tail(N)
    gf = h['home_goals'].sum() + a['away_goals'].sum()
    ga = h['away_goals'].sum() + a['home_goals'].sum()
    games = len(h)+len(a)
    pts = 0
    for _, r in h.iterrows():
        if r['home_goals']>r['away_goals']: pts+=3
        elif r['home_goals']==r['away_goals']: pts+=1
    for _, r in a.iterrows():
        if r['away_goals']>r['home_goals']: pts+=3
        elif r['away_goals']==r['home_goals']: pts+=1
    return {"gfpg": (gf/max(1,games)), "gapg": (ga/max(1,games)), "gdpg": ((gf-ga)/max(1,games)), "points": pts, "games": games}

def upset_probability(pH, pD, pA, elo_gap):
    fav = "H" if pH >= pA else "A"
    base_upset = pA if fav=="H" else pH
    extra = min(0.08, (elo_gap/400.0)*0.06)
    return min(1.0, base_upset + extra)

def analyze_today(question, cfg, hist_csv="data/matches_hist.csv", up_csv="data/upcoming.csv"):
    today = datetime.now(TZ).date()
    use_api = bool(cfg.get("football_data_org", {}).get("api_key"))
    if use_api:
        hist, up, _ = fetch_today_from_api(cfg)
        if hist is None or up is None:
            hist = pd.read_csv(hist_csv, parse_dates=['date']); up = pd.read_csv(up_csv, parse_dates=['date'])
    else:
        hist = pd.read_csv(hist_csv, parse_dates=['date']); up = pd.read_csv(up_csv, parse_dates=['date'])

    up_today = up[up['date'].dt.date == today].copy()
    if up_today.empty:
        return today.isoformat(), []

    elo = build_elo(hist)
    results = []
    for _, row in up_today.iterrows():
        home, away = row['home'], row['away']
        # Forma
        fh = recent_form(hist, home, N=6); fa = recent_form(hist, away, N=6)
        # Goles esperados (Elo + forma + localía)
        base_lambda = 1.35
        elo_diff = (elo.get(home,1500) - elo.get(away,1500))/100.0
        lam_h = max(0.05, base_lambda * np.exp(0.0035*elo_diff + 0.08*fh["gdpg"] + 0.15))
        lam_a = max(0.05, base_lambda * np.exp(-0.0035*elo_diff + 0.08*fa["gdpg"]))
        # Matriz Poisson
        M = match_score_matrix(lam_h, lam_a, max_goals=8)
        pH, pD, pA = probs_1x2_from_matrix(M)
        pOver25 = prob_over_from_matrix(M, 2.5)
        # Sorpresa
        gap = abs(elo.get(home,1500)-elo.get(away,1500))
        pUpset = upset_probability(pH, pD, pA, gap)
        # Doble oportunidad
        dc = {"1X": pH+pD, "12": pH+pA, "X2": pD+pA}
        best_dc = max(dc.items(), key=lambda x: x[1])
        # Corners (aprox)
        def tcm(team, side):
            if side=="home":
                return hist[hist['home']==team]['home_corners'].tail(6).mean()
            return hist[hist['away']==team]['away_corners'].tail(6).mean()
        ch = tcm(home,"home"); ca = tcm(away,"away")
        lam_corners = max(6.0, (0 if pd.isna(ch) else ch) + (0 if pd.isna(ca) else ca))
        from math import factorial
        def pmf(k, lam): return np.exp(-lam)*lam**k/factorial(k)
        pCornersOver = 1 - sum(pmf(i, lam_corners) for i in range(10))

        results.append({
            "league": row.get("league",""),
            "match": f"{home} vs {away}",
            "p_home": round(float(pH),3),
            "p_draw": round(float(pD),3),
            "p_away": round(float(pA),3),
            "p_over25": round(float(pOver25),3),
            "p_corners_over95": round(float(pCornersOver),3),
            "p_upset": round(float(pUpset),3),
            "double_chance_best": {"market": best_dc[0], "p": round(float(best_dc[1]),3)},
        })
    return today.isoformat(), results
