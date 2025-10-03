import argparse, requests, yaml, pandas as pd

SPORT_KEY = "soccer"

def fetch_odds(api_key, base_url, region="eu", markets=("h2h","totals")):
    url = f"{base_url}/sports/{SPORT_KEY}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": ",".join(markets)}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    data = r.json()
    rows = []
    for game in data:
        date = game["commence_time"][:10]
        home = game["home_team"]; away = game["away_team"]
        league = game.get("sport_title","Soccer")
        if not game.get("bookmakers"): 
            continue
        bk = game["bookmakers"][0]
        h2h = next((m for m in bk["markets"] if m["key"]=="h2h"), None)
        totals = next((m for m in bk["markets"] if m["key"]=="totals"), None)
        row = {"date": date, "league": league, "home": home, "away": away,
               "home_odds": None, "draw_odds": None, "away_odds": None,
               "ou25_over_odds": None, "ou25_under_odds": None}
        if h2h:
            def to_decimal(price):
                if isinstance(price, (int,float)):
                    return 1 + (price/100) if price > 0 else 1 + (100/abs(price))
                return None
            for oc in h2h["outcomes"]:
                if oc["name"]==home: row["home_odds"] = to_decimal(oc["price"])
                elif oc["name"]==away: row["away_odds"] = to_decimal(oc["price"])
                elif oc["name"] in ("Draw","Tie"): row["draw_odds"] = to_decimal(oc["price"])
        if totals:
            for oc in totals["outcomes"]:
                if float(oc.get("point", 0))==2.5:
                    if oc["name"].lower().startswith("over"):
                        row["ou25_over_odds"] = 1 + (oc["price"]/100) if isinstance(oc["price"], (int,float)) else None
                    else:
                        row["ou25_under_odds"] = 1 + (oc["price"]/100) if isinstance(oc["price"], (int,float)) else None
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/api_keys.yaml")
    p.add_argument("--upcoming_in", default="data/upcoming.csv")
    p.add_argument("--upcoming_out", default="data/upcoming.csv")
    p.add_argument("--region", default="eu")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    odds = fetch_odds(cfg["the_odds_api"]["api_key"], cfg["the_odds_api"]["base_url"], region=args.region)
    up = pd.read_csv(args.upcoming_in, parse_dates=["date"])
    m = up.merge(odds, on=["date","league","home","away"], how="left", suffixes=("","_odds"))
    for c in ["home_odds","draw_odds","away_odds","ou25_over_odds","ou25_under_odds"]:
        m[c] = m[c+"_odds"].combine_first(m[c])
        m.drop(columns=[c+"_odds"], inplace=True)
    m.to_csv(args.upcoming_out, index=False)
    print(f"Unidas odds -> {args.upcoming_out} ({len(odds)} filas de odds)") 
print('fetch odds placeholder')
