# src/ask.py
import logging
import yaml, pandas as pd, numpy as np, requests
from datetime import datetime, timedelta
import pytz
from elo import build_elo
from ensemble import ensemble_predict  # <— ENSAMBLE (Poisson + Dixon–Coles light + Elo/BT)

TZ = pytz.timezone("America/Bogota")

def load_config(path="config/api_keys.yaml"):
    try:
        return yaml.safe_load(open(path))
    except Exception:
        return {}

def api_get(url, headers=None, params=None, timeout=25):
    """Wrapper con logging de errores para football-data.org"""
    r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
    if r.status_code != 200:
        logging.error("FD.org %s %s -> %s %s", url, params, r.status_code, r.text[:200])
    r.raise_for_status()
    return r.json()

def fetch_today_from_api(cfg):
    """
    Descarga histórico (últimos ~180 días) y HOY desde football-data.org
    en ventanas de <=10 días para evitar 400 Bad Request.
    """
    api = cfg.get("football_data_org", {})
    token = api.get("api_key")
    comps = api.get("competitions", ["PL","PD","SA","BL1","FL1","DED","PPL"])
    if not token:
        return None, None, None

    from datetime import timedelta
    API_BASE = "https://api.football-data.org/v4"
    headers = {"X-Auth-Token": token}

    today = datetime.now(TZ).date()
    start = today - timedelta(days=180)

    hist_rows, up_rows = [], []
    window = timedelta(days=9)  # <=10 días

    for comp in comps:
        # HISTÓRICO en ventanas
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
            except Exception:
                # si una ventana falla, seguimos con la siguiente
                pass
            d = to + timedelta(days=1)

        # PARTIDOS DE HOY
        try:
            m_today = api_get(
                f"{API_BASE}/competitions/{comp}/matches",
                headers,
                {"dateFrom": today.isoformat(), "dateTo": today.isoformat()},
            )
            for match in m_today.get("matches", []):
                if match.get("status") not in ("TIMED", "SCHEDULED"):
                    continue
                up_rows.append({
                    "date": (match.get("utcDate","")[:10]),
                    "league": m_today.get("competition",{}).get("name", comp),
                    "home": match.get("homeTeam",{}).get("name"),
                    "away": match.get("awayTeam",{}).get("name"),
                    "home_odds": None, "draw_odds": None, "away_odds": None,
                    "ou25_over_odds": None, "ou25_under_odds": None,
                })
        except Exception:
            pass

    return pd.DataFrame(hist_rows), pd.DataFrame(up_rows), pd.DataFrame()

def recent_form(df_hist, team, N=6):
    """Forma reciente simple: goles a favor/contra y puntos en últimos N juegos."""
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
    """Heurística para 'sorpresa': prob de que gane el no favorito + ajuste por gap Elo."""
    fav = "H" if pH >= pA else "A"
    base_upset = pA if fav=="H" else pH
    extra = min(0.08, (elo_gap/400.0)*0.06)
    return min(1.0, base_upset + extra)

def analyze_today(question, cfg, hist_csv="data/matches_hist.csv", up_csv="data/upcoming.csv"):
    """
    Lógica principal:
    - Usa API (si hay token) para histórico y partidos de HOY; si falla, usa CSVs.
    - Calcula Elo + forma, lambdas de gol y corre por ENSAMBLE.
    - Devuelve lista de estudios listos para Telegram.
    """
    today = datetime.now(TZ).date()
    use_api = bool(cfg.get("football_data_org", {}).get("api_key"))
    if use_api:
        hist, up, _ = fetch_today_from_api(cfg)
        if hist is None or up is None:
            hist = pd.read_csv(hist_csv, parse_dates=['date'])
            up = pd.read_csv(up_csv, parse_dates=['date'])
    else:
        hist = pd.read_csv(hist_csv, parse_dates=['date'])
        up = pd.read_csv(up_csv, parse_dates=['date'])

    up_today = up[up['date'].dt.date == today].copy()
    if up_today.empty:
        return today.isoformat(), []

    elo = build_elo(hist)
    results = []
    for _, row in up_today.iterrows():
        home, away = row['home'], row['away']

        # Forma reciente
        fh = recent_form(hist, home, N=6)
        fa = recent_form(hist, away, N=6)

        # Lambdas base (Elo + forma + localía)
        base_lambda = 1.35
        elo_diff = (elo.get(home,1500) - elo.get(away,1500))/100.0
        lam_h = max(0.05, base_lambda * np.exp(0.0035*elo_diff + 0.08*fh["gdpg"] + 0.15))
        lam_a = max(0.05, base_lambda * np.exp(-0.0035*elo_diff + 0.08*fa["gdpg"]))

        # === ENSAMBLE ===
        ens = ensemble_predict(
            lam_h, lam_a,
            elo.get(home,1500), elo.get(away,1500),
            weights=(0.4, 0.35, 0.25),  # Poisson / Dixon–Coles / Elo-BT
            rho=0.12
        )
        pH, pD, pA = ens["pH"], ens["pD"], ens["pA"]
        pOver25 = ens["pOver25"]

        # Sorpresa
        gap = abs(elo.get(home,1500)-elo.get(away,1500))
        pUpset = upset_probability(pH, pD, pA, gap)

        # Doble oportunidad
        dc = {"1X": pH+pD, "12": pH+pA, "X2": pD+pA}
        best_dc = max(dc.items(), key=lambda x: x[1])

        # Corners (aprox por media de los últimos juegos si existen)
        def mean_last_corners(team, side):
            if side=="home":
                return hist[hist['home']==team]['home_corners'].tail(6).mean()
            return hist[hist['away']==team]['away_corners'].tail(6).mean()
        ch = mean_last_corners(home,"home"); ca = mean_last_corners(away,"away")
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
            "models": ens["components"],  # detalle por modelo (útil para depurar/explicar)
        })
    return today.isoformat(), results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*", help="Ej: 'qué apuestas hay para hoy?'")
    args = p.parse_args()
    cfg = load_config()
    date_str, res = analyze_today(" ".join(args.question) if args.question else "hoy", cfg)
    print(f"Estudio de partidos para {date_str}:")
    for r in res:
        print(f"- {r['league']}: {r['match']}  (1X2: {r['p_home']}/{r['p_draw']}/{r['p_away']}, Over2.5={r['p_over25']}, Corners>9.5={r['p_corners_over95']}, Upset={r['p_upset']})")
