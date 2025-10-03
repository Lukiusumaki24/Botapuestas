import argparse, requests, yaml, pandas as pd
from datetime import datetime, timedelta, timezone

def fetch_fixtures(api_key, base_url, league_ids, days_back=120, days_forward=7):
    headers = {"x-apisports-key": api_key}
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_back)
    end = today + timedelta(days=days_forward)
    rows_hist, rows_upcoming = [], []

    for lid in league_ids:
        url = f"{base_url}/fixtures"
        params = {"league": lid, "from": start.isoformat(), "to": today.isoformat()}
        r = requests.get(url, headers=headers, params=params, timeout=30); r.raise_for_status()
        for x in r.json().get("response", []):
            goal = x.get("goals") or {}
            rows_hist.append({
                "date": (x["fixture"]["date"] or "")[:10],
                "league": x["league"]["name"],
                "home": x["teams"]["home"]["name"],
                "away": x["teams"]["away"]["name"],
                "home_goals": goal.get("home"),
                "away_goals": goal.get("away"),
                "home_corners": None, "away_corners": None
            })
        params = {"league": lid, "from": today.isoformat(), "to": end.isoformat()}
        r = requests.get(url, headers=headers, params=params, timeout=30); r.raise_for_status()
        for x in r.json().get("response", []):
            rows_upcoming.append({
                "date": (x["fixture"]["date"] or "")[:10],
                "league": x["league"]["name"],
                "home": x["teams"]["home"]["name"],
                "away": x["teams"]["away"]["name"],
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
    hist, up = fetch_fixtures(cfg["api_football"]["api_key"], cfg["api_football"]["base_url"], cfg["api_football"]["leagues"])
    hist.to_csv(args.hist_out, index=False)
    up.to_csv(args.upcoming_out, index=False)
    print(f"Histórico: {len(hist)} filas -> {args.hist_out}")
    print(f"Próximos: {len(up)} filas -> {args.upcoming_out}")
print('fetch fixtures API-Football placeholder')
