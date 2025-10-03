import os, logging, pytz, re
from datetime import datetime, timedelta
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters
from ask import analyze_today, load_config

TZ = pytz.timezone("America/Bogota")
HELP_TEXT = (
    "Hola 👋 Soy tu bot de pronósticos.\n\n"
    "Pregúntame:\n"
    "• apuestas para hoy\n"
    "• apuestas para mañana\n"
    "• apuestas para 05/10\n\n"
    "Yo haré el estudio (forma, H2H, localía, lesiones* si hay, sorpresa, goles y corners).\n"
    "Salida: 1X2, Doble Oportunidad, Over/Under 2.5, Corners >9.5, prob. de upset.\n"
    "*Lesiones dependen de la API configurada."
)

def format_reply(date_str: str, results: list) -> str:
    if not results:
        return f"No encontré partidos para {date_str}."
    lines = [f"*Estudio de partidos para {date_str}* (America/Bogota)"]
    for r in results[:20]:
        lines.append(
            f"\n*{r.get('league','')}*: {r['match']}\n"
            f"• 1X2: Local *{r['p_home']}*, Empate *{r['p_draw']}*, Visita *{r['p_away']}*\n"
            f"• Doble Oportunidad: *{r['double_chance_best']['market']}* (p≈{r['double_chance_best']['p']})\n"
            f"• Goles: Over2.5 p≈{r['p_over25']} | Corners>9.5 p≈{r['p_corners_over95']} | Upset p≈{r['p_upset']}\n"
        )
    if len(results) > 20:
        lines.append(f"\n… y {len(results)-20} partidos más.")
    return "\n".join(lines)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.message.text or ""
    cfg = load_config()
    date_str, res = analyze_today(q, cfg, hist_csv="data/matches_hist.csv", up_csv="data/upcoming.csv")
    reply = format_reply(date_str, res)
    await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)

if __name__ == "__main__":
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Falta TELEGRAM_BOT_TOKEN en variables de entorno.")
    logging.basicConfig(level=logging.INFO)
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling()
print('Use chatbot.py in full build')
