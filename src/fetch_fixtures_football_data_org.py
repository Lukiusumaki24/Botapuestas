import argparse, requests, yaml, pandas as pd
from datetime import datetime, timedelta, timezone

API_BASE = "https://api.football-data.org/v4"

def fd_get(path, token, params=None):
    headers = {"X-Auth-Token": token}
    r = requests.get(f"{API_BASE}{path}", headers=headers, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch(fd_token, competitions, days_back=120, days_forward=7):
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_back)
    end = today + timedelta(days=days_forward)
    rows_hist, rows_upcoming = [], []

    for comp in competitions:
        matches = fd_get(f"/competitions/{comp}/matches", fd_token, {"dateFrom": start.isoformat(), "dateTo": today.isoformat()})
        for m in matches.get("matches", []):
            if m.get("status") != "FINISHED":
                continue
            score = m.get("score",{}).get("fullTime",{})
            rows_hist.append({
                "date": m.get("utcDate","")[:10],
                "league": matches.get("competition",{}).get("name", comp),
                "home": m.get("homeTeam",{}).get("name"),
                "away": m.get("awayTeam",{}).get("name"),
                "home_goals": score.get("home", None),
                "away_goals": score.get("away", None),
                "home_corners": None,
                "away_corners": None,
            })

        upcoming = fd_get(f"/competitions/{comp}/matches", fd_token, {"dateFrom": today.isoformat(), "dateTo": end.isoformat()})
        for m in upcoming.get("matches", []):
            if m.get("status") not in ("TIMED","SCHEDULED"):
                continue
            rows_upcoming.append({
                "date": m.get("utcDate","")[:10],
                "league": upcoming.get("competition",{}).get("name", comp),
                "home": m.get("homeTeam",{}).get("name"),
                "away": m.get("awayTeam",{}).get("name"),
                "home_odds": None, "draw_odds": None, "away_odds": None,
                "ou25_over_odds": None, "ou25_under_odds": None
            })

    return pd.DataFrame(rows_hist), pd.DataFrame(rows_upcoming)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/api_keys.yaml")
    p.add_argument("--hist_out", default="data/matches_hist.csv")
    p.add_argument("--upcoming_out", default="data/upcoming.csv")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    token = cfg.get("football_data_org", {}).get("api_key")
    if not token:
        raise SystemExit("Falta api_key en football_data_org en config/api_keys.yaml")
    comps = cfg.get("football_data_org", {}).get("competitions", ["PL","PD","SA","BL1","FL1","DED","PPL"])
    hist, up = fetch(token, comps)
    hist.to_csv(args.hist_out, index=False)
    up.to_csv(args.upcoming_out, index=False)
    print(f"Histórico: {len(hist)} filas -> {args.hist_out}")
    print(f"Próximos: {len(up)} filas -> {args.upcoming_out}")
print('fetch fixtures football-data.org placeholder')
