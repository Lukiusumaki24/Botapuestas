#!/usr/bin/env bash
set -euo pipefail

# genera config desde variables de entorno (si están)
mkdir -p config data
cat > config/api_keys.yaml <<'YAML'
football_data_org:
  api_key: "${FOOTBALL_DATA_API_KEY:-}"
  competitions: ["PL","PD","SA","BL1","FL1","DED","PPL","ELC","BSA","CL","EC","WC"]

the_odds_api:
  api_key: "${ODDS_API_KEY:-}"
  base_url: "https://api.the-odds-api.com/v4"
YAML


# inicia bot de telegram vía webhook (escucha en $PORT y registra /webhook)
python src/telegram_web.py
