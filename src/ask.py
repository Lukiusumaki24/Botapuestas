import os
import logging
from datetime import datetime, timedelta
import pytz
import yaml
import pandas as pd
import numpy as np
import requests
from math import exp, factorial

# --- módulos internos existentes en tu repo ---
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
    - FOOTBALL_DATA_API_KEY (env) > api_keys.yaml
    - USE_API=0 fuerza modo local aunque haya token
    """
    try:
        cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    except Exception:
        cfg = {}

    env_token = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if env_token:
        cfg.setdefault("football_data_org", {})
        cfg["football_data_org"]["api_key"] = env_token

    # bandera para desactivar API aunque exista token
    cfg["USE_API"] = os.getenv("USE_API", "0").strip()  # default 0 = local
    return cfg

def api_get(url, headers=None, params=None, timeout=25):
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

# =============== Helpers de Poisson básicos para el flujo /analizar5 ===============

def pois_pmf(lmbda: float, k: int) -> float:
    return (lmbda ** k) * exp(-lmbda) / factorial(k)

def match_probabilities_from_lambdas(lh: float, la: float, max_goals: int = 8):
    """
    Calcula p(H), p(D), p(A) sumando la matriz de Poisson hasta max_goals.
    También devuelve pOver25 para total > 2.
    """
    p_home = p_draw = p_away = 0.0
    p_total_gt2 = 0.0
    for gh in range(0, max_goals + 1):
        for ga in range(0, max_goals + 1):
            p = pois_pmf(lh, gh) * pois_pmf(la, ga)
            if gh > ga: p_home += p
            elif gh == ga: p_draw += p
            else: p_away += p
            if gh + ga > 2: p_total_gt2 += p
    # Normalizar pequeñas pérdidas numéricas
    s = p_home + p_draw + p_away
    if s > 0:
        p_home, p_draw, p_away = p_home/s, p_draw/s, p_away/s
    return p_home, p_draw, p_away, p_total_gt2

# ==================== Football-Data.org (se mantiene por compatibilidad) ====================

def fetch_today_from_api(cfg):
    api = cfg.get("football_data_org", {})
    token = (api.get("api_key") or "").strip()
    comps = api.get("competitions", ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL"])
    # Si USE_API=0 o no hay token → desactivar
    if cfg.get("USE_API", "0") == "0" or not token:
        return None, None, None

    API_BASE = "https://api.football-data.org/v4"
    headers = {"X-Auth-Token": token}

    today = datetime.now(TZ).date()
    start = today - timedelta(days=180)
    hist_rows, up_rows = [], []
    window = timedelta(days=9)

    # Sonda rápida para validar token
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

    for comp in comps:
        # (histórico y hoy) — igual que antes, omitido por brevedad
        pass  # en modo local no se usa; dejamos stub para compatibilidad

    return pd.DataFrame(hist_rows), pd.DataFrame(up_rows), pd.DataFrame()

# ==================== Modo “CSV local” clásico (apuestas para hoy) ====================

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

    # Forzar local si USE_API=0
    use_api = (cfg.get("USE_API", "0") == "1") and bool((cfg.get("football_data_org", {}) or {}).get("api_key"))
    if use_api:
        hist, up, _ = fetch_today_from_api(cfg)
        if hist is None or up is None or up.empty:
            hist = pd.read_csv(hist_csv)
            up = pd.read_csv(up_csv)
    else:
        hist = pd.read_csv(hist_csv)
        up = pd.read_csv(up_csv)

    # Asegurar dtype fecha
    for df in (hist, up):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.dropna(subset=["date"], inplace=True)

    # Asegurar columnas extendidas (histórico)
    cols_needed = [
        "home_corners","away_corners","home_shots","away_shots","home_shots_on","away_shots_on",
        "home_cards","away_cards","home_possession","away_possession","home_fouls","away_fouls","weather"
    ]
    for c in cols_needed:
        if c not in hist.columns:
            hist[c] = np.nan

    # Asegurar columnas nuevas (upcoming)
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

        fh = rolling_team_stats(hist, home, n=6)
        fa = rolling_team_stats(hist, away, n=6)

        base_lambda = 1.35
        elo_diff = (elo.get(home, 1500) - elo.get(away, 1500)) / 100.0

        climate_penalty = 0.0
        if "weather" in hist.columns:
            recent_weather = hist.sort_values("date").tail(30)["weather"].dropna().astype(str).str.lower()
            if len(recent_weather):
                if any(recent_weather.str.contains("rain")):
                    climate_penalty -= 0.03
                if any(recent_weather.str_contains("storm")):
                    climate_penalty -= 0.06

        lam_h = max(0.05, base_lambda * np.exp(0.0035 * elo_diff + 0.08 * (fh["gdpg"]) + 0.15 + climate_penalty))
        lam_a = max(0.05, base_lambda * np.exp(-0.0035 * elo_diff + 0.08 * (fa["gdpg"]) + climate_penalty))

        ens = ensemble_predict(lam_h, lam_a, elo.get(home, 1500), elo.get(away, 1500),
                               weights=(0.45, 0.35, 0.20), rho=0.12)
        pH, pD, pA = ens["pH"], ens["pD"], ens["pA"]
        pOver25 = ens["pOver25"]

        lam_corners = combine_home_away_rate(fh.get("corners_for_pg"), fa.get("corners_for_pg"))
        pCornersOver = poisson_over_prob(lam_corners, line=9.5, max_k=35)

        lam_cards = max(0.5, (fh.get("cards_for_pg") or 0) + (fa.get("cards_for_pg") or 0))
        pCardsOver45 = poisson_over_prob(lam_cards, line=4.5, max_k=25)

        p_1x = float(pH + pD)
        p_x2 = float(pD + pA)
        p_12 = float(pH + pA)
        dc = {"1X": p_1x, "12": p_12, "X2": p_x2}
        best_dc = max(dc.items(), key=lambda x: x[1])

        gap = abs(elo.get(home, 1500) - elo.get(away, 1500))
        pUpset = upset_probability(pH, pD, pA, gap)

        markets = []  # en modo local sin cuotas, no calculamos EV/Kelly
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
            "double_chance_best": {"market": best_dc[0], "p": round(float(best_dc[1]), 3)},
            "models": ens["components"],
            "markets": markets,
            "value_bets": [],  # sin cuotas
        })

    return target_date.isoformat(), results

# ==================== NUEVO: análisis rápido desde últimos 5 resultados ====================

def analyze_from_recent(home: str, away: str,
                        home_last5: list[tuple[int, int]],
                        away_last5: list[tuple[int, int]],
                        home_advantage: float = 0.15):
    """
    Recibe:
      - home, away: nombres de equipos
      - home_last5: lista de 5 tuplas (gf, gc) del equipo local en sus últimos partidos
      - away_last5: lista de 5 tuplas (gf, gc) del visitante
    Calcula lambdas Poisson y devuelve probabilidades de 1X2, Over2.5, DC y upset.
    """

    # Medias simples de goles
    h_for = np.mean([gf for gf, gc in home_last5]) if home_last5 else 1.2
    h_against = np.mean([gc for gf, gc in home_last5]) if home_last5 else 1.2
    a_for = np.mean([gf for gf, gc in away_last5]) if away_last5 else 1.2
    a_against = np.mean([gc for gf, gc in away_last5]) if away_last5 else 1.2

    # Mezcla ofensiva/defensiva + ventaja de local
    lam_h = max(0.05, 0.5 * h_for + 0.5 * a_against + home_advantage)
    lam_a = max(0.05, 0.5 * a_for + 0.5 * h_against)

    pH, pD, pA, pOver25 = match_probabilities_from_lambdas(lam_h, lam_a, max_goals=8)

    # Doble oportunidad
    p_1x = float(pH + pD)
    p_x2 = float(pD + pA)
    p_12 = float(pH + pA)
    dc = {"1X": p_1x, "12": p_12, "X2": p_x2}
    best_dc = max(dc.items(), key=lambda x: x[1])

    # Upset simple: no favorito gana
    fav = "H" if pH >= pA else "A"
    base_upset = pA if fav == "H" else pH
    pUpset = float(base_upset)

    result = {
        "league": "",
        "match": f"{home} vs {away}",
        "lambdas": {"home": round(lam_h, 3), "away": round(lam_a, 3)},
        "p_home": round(float(pH), 3),
        "p_draw": round(float(pD), 3),
        "p_away": round(float(pA), 3),
        "p_over25": round(float(pOver25), 3),
        "double_chance_best": {"market": best_dc[0], "p": round(float(best_dc[1]), 3)},
        "p_upset": round(float(pUpset), 3),
    }
    return result
