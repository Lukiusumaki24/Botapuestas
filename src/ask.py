import os
import logging
from datetime import datetime, timedelta
import pytz
import yaml
import pandas as pd
import numpy as np
import requests

# --- tus módulos internos (ya en el repo) ---
from elo import build_elo
from features import rolling_team_stats
from ensemble import ensemble_predict
from models.stats import poisson_over_prob, combine_home_away_rate

TZ = pytz.timezone("America/Bogota")

# ==================== Staking ====================
KELLY_MULTIPLIER = 1.0  # 1.0 = Kelly completo; 0.5 = medio Kelly
KELLY_CAP = 0.25        # límite del 25% del bank

# ==================== Utilidades ====================

class InvalidAPIToken(Exception):
    """Señaliza token inválido/403 para hacer fallback inmediato a CSV."""
    pass

def load_config(path="config/api_keys.yaml"):
    """
    Carga config desde YAML y prioriza variables de entorno.
    FOOTBALL_DATA_API_KEY (env) > api_keys.yaml
    """
    try:
        cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    except Exception:
        cfg = {}

    env_token = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if env_token:
        cfg.setdefault("football_data_org", {})
        cfg["football_data_org"]["api_key"] = env_token

    return cfg

def api_get(url, headers=None, params=None, timeout=25):
    """GET con detección de token inválido para cortar rápido."""
    r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
    if r.status_code in (400, 401, 403):
        msg = (r.text or "").strip()[:200]
        raise InvalidAPIToken(f"{r.status_code} {url} :: {msg}")
    if r.status_code != 200:
        logging.error("HTTP %s -> %s | %s", r.status_code, url, (r.text[:200] if r.text else ""))
    r.raise_for_status()
    return r.json()

def kelly_fraction(p: float, odds: float, multiplier: float = KELLY_MULTIPLIER, cap: float = KELLY_CAP) -> float:
    if odds is None or odds <= 1.0 or p is None or p <= 0 or p >= 1:
        return 0.0
    b = odds - 1.0
    f = (b * p - (1 - p)) / b
    f = max(0.0, f)
    f = min(f, cap)
    return round(float(f * multiplier), 4)

def expected_value(p: float, odds: float) -> float:
    if odds is None or odds <= 1.0 or p is None:
        return None
    return round(float(p * odds - 1.0), 4)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# ==================== Football-Data.org ====================

def fetch_today_from_api(cfg):
    """
    Descarga histórico (180d, ventanas de 10d) y partidos de hoy
    para las competiciones indicadas. Si el token es inválido o hay
    error de red, devuelve (None, None, None) para fallback a CSV.
    """
    api = cfg.get("football_data_org", {})
    token = (api.get("api_key") or "").strip()
    comps = api.get("competitions", ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL"])
    if not token:
        return None, None, None

    API_BASE = "https://api.football-data.org/v4"
    headers = {"X-Auth-Token": token}

    today = datetime.now(TZ).date()
    start = today - timedelta(days=180)
    hist_rows, up_rows = [], []
    window = timedelta(days=9)

    # Sonda rápida: si falla, hacemos fallback sin spamear logs
    try:
        _ = api_get(
            f"{API_BASE}/competitions/PL/matches",
            headers,
            {"dateFrom": today.isoformat(), "dateTo": today.isoformat()},
        )
    except InvalidAPIToken as e:
        logging.warning("API token inválido o sin permisos. Fallback a CSV. %s", e)
        return None, None, None
    except Exception:
        logging.warning("Error de red con la API. Fallback a CSV.")
        return None, None, None

    # Si llegamos aquí, el token sirve. Traemos datos por competición.
    for comp in comps:
        # Histórico por ventanas
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
                        "date": (match.get("utcDate", "")[:10]),
                        "league": m.get("competition", {}).get("name", comp),
                        "home": match.get("homeTeam", {}).get("name"),
                        "away": match.get("awayTeam", {}).get("name"),
                        "home_goals": sc.get("home", None),
                        "away_goals": sc.get("away", None),
                        # columnas extendidas (no las da la API free)
                        "home_corners": None, "away_corners": None,
                        "home_shots": None, "away_shots": None,
                        "home_shots_on": None, "away_shots_on": None,
                        "home_cards": None, "away_cards": None,
                        "home_possession": None, "away_possession": None,
                        "home_fouls": None, "away_fouls": None,
                        "weather": None,
                    })
            except InvalidAPIToken:
                logging.warning("Token inválido durante histórico. Fallback a CSV.")
                return None, None, None
            except Exception:
                # errores de red puntuales → seguimos
                pass
            d = to + timedelta(days=1)

        # Partidos de hoy
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
                    "date": (match.get("utcDate", "")[:10]),
                    "league": m_today.get("competition", {}).get("name", comp),
                    "home": match.get("homeTeam", {}).get("name"),
                    "away": match.get("awayTeam", {}).get("name"),
                    "home_odds": None, "draw_odds": None, "away_odds": None,
                    "ou25_over_odds": None, "ou25_under_odds": None,
                    "corners_over95_odds": None, "cards_over45_odds": None,
                    "dc_1x_odds": None, "dc_x2_odds": None, "dc_12_odds": None,
                })
        except InvalidAPIToken:
            logging.warning("Token inválido durante HOY. Fallback a CSV.")
            return None, None, None
        except Exception:
            pass

    return pd.DataFrame(hist_rows), pd.DataFrame(up_rows), pd.DataFrame()

# ==================== Análisis ====================

def upset_probability(pH, pD, pA, elo_gap):
    fav = "H" if pH >= pA else "A"
    base_upset = pA if fav == "H" else pH
    extra = min(0.08, (elo_gap / 400.0) * 0.06)
    return float(min(1.0, base_upset + extra))

def parse_when_from_question(q: str):
    q = (q or "").lower()
    today = datetime.now(TZ).date()
    if "mañana" in q:
        return today + timedelta(days=1)
    if "hoy" in q:
        return today
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m", "%d-%m"):
        try:
            dt = datetime.strptime(q.strip(), fmt)
            if fmt in ("%d/%m", "%d-%m"):
                return dt.replace(year=today.year).date()
            return dt.date()
        except Exception:
            pass
    return today

def analyze_today(question, cfg, hist_csv="data/matches_hist.csv", up_csv="data/upcoming.csv"):
    target_date = parse_when_from_question(question)

    use_api = bool((cfg.get("football_data_org", {}) or {}).get("api_key"))
    if use_api:
        hist, up, _ = fetch_today_from_api(cfg)
        if hist is None or up is None or up.empty:
            hist = pd.read_csv(hist_csv)
            up = pd.read_csv(up_csv)
    else:
        hist = pd.read_csv(hist_csv)
        up = pd.read_csv(up_csv)

    # --- asegurar dtype fecha para evitar errores .dt ---
    for df in (hist, up):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.dropna(subset=["date"], inplace=True)

    # --- asegurar columnas extendidas (histórico) ---
    cols_needed = [
        "home_corners","away_corners","home_shots","away_shots","home_shots_on","away_shots_on",
        "home_cards","away_cards","home_possession","away_possession","home_fouls","away_fouls","weather"
    ]
    for c in cols_needed:
        if c not in hist.columns:
            hist[c] = np.nan

    # --- asegurar columnas nuevas (upcoming) ---
    for c in ["corners_over95_odds","cards_over45_odds","dc_1x_odds","dc_x2_odds","dc_12_odds"]:
        if c not in up.columns:
            up[c] = np.nan

    up_target = up[up["date"].dt.date == target_date].copy()
    if up_target.empty:
        return target_date.isoformat(), []

    # Elo con histórico propio
    elo = build_elo(hist)

    results = []
    for _, row in up_target.iterrows():
        home, away = row["home"], row["away"]

        # Stats recientes (6 partidos)
        fh = rolling_team_stats(hist, home, n=6)
        fa = rolling_team_stats(hist, away, n=6)

        # Lambdas gol: Elo + forma + localía + clima
        base_lambda = 1.35
        elo_diff = (elo.get(home, 1500) - elo.get(away, 1500)) / 100.0

        climate_penalty = 0.0
        if "weather" in hist.columns:
            recent_weather = hist.sort_values("date").tail(30)["weather"].dropna().astype(str).str.lower()
            if len(recent_weather):
                if any(recent_weather.str.contains("rain")):
                    climate_penalty -= 0.03
                if any(recent_weather.str.contains("storm")):
                    climate_penalty -= 0.06

        lam_h = max(0.05, base_lambda * np.exp(0.0035 * elo_diff + 0.08 * (fh["gdpg"]) + 0.15 + climate_penalty))
        lam_a = max(0.05, base_lambda * np.exp(-0.0035 * elo_diff + 0.08 * (fa["gdpg"]) + climate_penalty))

        # Ensamble (Poisson + Dixon-Coles light + Elo/BT)
        ens = ensemble_predict(lam_h, lam_a, elo.get(home, 1500), elo.get(away, 1500),
                               weights=(0.45, 0.35, 0.20), rho=0.12)
        pH, pD, pA = ens["pH"], ens["pD"], ens["pA"]
        pOver25 = ens["pOver25"]

        # Corners > 9.5
        lam_corners = combine_home_away_rate(fh.get("corners_for_pg"), fa.get("corners_for_pg"))
        pCornersOver = poisson_over_prob(lam_corners, line=9.5, max_k=35)

        # Tarjetas > 4.5
        lam_cards = max(0.5, (fh.get("cards_for_pg") or 0) + (fa.get("cards_for_pg") or 0))
        pCardsOver45 = poisson_over_prob(lam_cards, line=4.5, max_k=25)

        # Doble oportunidad (prob propia)
        p_1x = float(pH + pD)
        p_x2 = float(pD + pA)
        p_12 = float(pH + pA)
        dc = {"1X": p_1x, "12": p_12, "X2": p_x2}
        best_dc = max(dc.items(), key=lambda x: x[1])

        # Sorpresa (gap Elo)
        gap = abs(elo.get(home, 1500) - elo.get(away, 1500))
        pUpset = upset_probability(pH, pD, pA, gap)

        # EV & Kelly si hay cuotas
        home_odds = safe_float(row.get("home_odds"))
        draw_odds = safe_float(row.get("draw_odds"))
        away_odds = safe_float(row.get("away_odds"))
        ou25_over_odds = safe_float(row.get("ou25_over_odds"))
        corners_over95_odds = safe_float(row.get("corners_over95_odds"))
        cards_over45_odds = safe_float(row.get("cards_over45_odds"))
        dc_1x_odds = safe_float(row.get("dc_1x_odds"))
        dc_x2_odds = safe_float(row.get("dc_x2_odds"))
        dc_12_odds = safe_float(row.get("dc_12_odds"))

        markets = []

        if home_odds:
            markets.append({"market": "1", "prob": round(pH, 3), "odds": home_odds,
                            "ev": expected_value(pH, home_odds), "kelly": kelly_fraction(pH, home_odds)})
        if draw_odds:
            markets.append({"market": "X", "prob": round(pD, 3), "odds": draw_odds,
                            "ev": expected_value(pD, draw_odds), "kelly": kelly_fraction(pD, draw_odds)})
        if away_odds:
            markets.append({"market": "2", "prob": round(pA, 3), "odds": away_odds,
                            "ev": expected_value(pA, away_odds), "kelly": kelly_fraction(pA, away_odds)})

        if ou25_over_odds:
            markets.append({"market": "Over 2.5", "prob": round(pOver25, 3), "odds": ou25_over_odds,
                            "ev": expected_value(pOver25, ou25_over_odds), "kelly": kelly_fraction(pOver25, ou25_over_odds)})

        if corners_over95_odds:
            markets.append({"market": "Corners > 9.5", "prob": round(pCornersOver, 3), "odds": corners_over95_odds,
                            "ev": expected_value(pCornersOver, corners_over95_odds), "kelly": kelly_fraction(pCornersOver, corners_over95_odds)})

        if cards_over45_odds:
            markets.append({"market": "Tarjetas > 4.5", "prob": round(pCardsOver45, 3), "odds": cards_over45_odds,
                            "ev": expected_value(pCardsOver45, cards_over45_odds), "kelly": kelly_fraction(pCardsOver45, cards_over45_odds)})

        if dc_1x_odds:
            markets.append({"market": "Doble Oport 1X", "prob": round(p_1x, 3), "odds": dc_1x_odds,
                            "ev": expected_value(p_1x, dc_1x_odds), "kelly": kelly_fraction(p_1x, dc_1x_odds)})
        if dc_x2_odds:
            markets.append({"market": "Doble Oport X2", "prob": round(p_x2, 3), "odds": dc_x2_odds,
                            "ev": expected_value(p_x2, dc_x2_odds), "kelly": kelly_fraction(p_x2, dc_x2_odds)})
        if dc_12_odds:
            markets.append({"market": "Doble Oport 12", "prob": round(p_12, 3), "odds": dc_12_odds,
                            "ev": expected_value(p_12, dc_12_odds), "kelly": kelly_fraction(p_12, dc_12_odds)})

        value_bets = [m for m in markets if (m.get("ev") is not None and m["ev"] > 0)]
        value_bets = sorted(value_bets, key=lambda x: x["ev"], reverse=True)

        results.append({
            "league": row.get("league", ""),
            "match": f"{home} vs {away}",
            "p_home": round(float(pH), 3),
            "p_draw": round(float(pD), 3),
            "p_away": round(float(pA), 3),
            "p_over25": round(float(pOver25), 3),
            "p_corners_over95": round(float(pCornersOver), 3),
            "p_cards_over45": round(float(pCardsOver45), 3),
            "p_upset": round(float(pUpset), 3),
            "double_chance_best": {"market": max(dc, key=dc.get), "p": round(float(max(dc.values())), 3)},
            "models": ens["components"],
            "markets": markets,
            "value_bets": value_bets[:5],
        })

    return target_date.isoformat(), results
