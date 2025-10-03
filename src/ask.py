def load_config(path='config/api_keys.yaml'):
    return {}

def analyze_today(q,cfg,hist_csv='data/matches_hist.csv',up_csv='data/upcoming.csv'):
    import pandas as pd
    from datetime import date
    try:
        up=pd.read_csv(up_csv,parse_dates=['date'])
        up_today=up[up['date'].dt.date==date.today()]
    except Exception:
        up_today=pd.DataFrame()
    res=[]
    for _,r in up_today.iterrows():
        res.append({'league':r.get('league',''),'match':f"{r['home']} vs {r['away']}",
        'p_home':0.45,'p_draw':0.27,'p_away':0.28,'p_over25':0.52,'p_corners_over95':0.55,'p_upset':0.22,
        'double_chance_best':{'market':'1X','p':0.72}})
    return str(date.today()),res
