import os, logging, pytz
import pandas as pd
from datetime import datetime
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    ConversationHandler, filters
)
from ask import analyze_today, load_config

# ==================== Config general ====================
TZ = pytz.timezone("America/Bogota")
HELP_TEXT = (
    "Hola 👋 Soy tu bot de pronósticos.\n\n"
    "Comandos:\n"
    "• /start → ayuda\n"
    "• /listar_hoy → muestra lo que hay en upcoming.csv para HOY\n"
    "• /cargar → ingresar un partido (CSV de próximos)\n"
    "• /resultado → ingresar resultado + estadísticas (CSV histórico)\n\n"
    "También puedes escribir:\n"
    "• 'apuestas para hoy'\n"
    "• 'apuestas para mañana'\n"
    "• 'apuestas para 2025-10-03'\n\n"
    "Analizo forma, Elo, Poisson/Dixon–Coles, localía, clima, corners y tarjetas.\n"
    "Salida: 1X2, Doble Oportunidad, Over/Under 2.5, Corners>9.5, Tarjetas>4.5, prob. de upset, y *value bets* con EV/Kelly."
)

UPCOMING_CSV = "data/upcoming.csv"
UPCOMING_COLS = [
    "date","league","home","away",
    "home_odds","draw_odds","away_odds",
    "ou25_over_odds","ou25_under_odds",
    "corners_over95_odds","cards_over45_odds",
    "dc_1x_odds","dc_x2_odds","dc_12_odds"
]

HIST_CSV = "data/matches_hist.csv"
HIST_COLS = [
    "date","league","home","away",
    "home_goals","away_goals",
    "home_corners","away_corners",
    "home_shots","away_shots",
    "home_shots_on","away_shots_on",
    "home_cards","away_cards",
    "home_possession","away_possession",
    "home_fouls","away_fouls",
    "weather",
]

def ensure_csv(path: str, columns: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)

def parse_float_or_none(txt: str):
    t = (txt or "").strip()
    if t == "" or t.lower() in ("na","none","null","-"):
        return None
    t = t.replace(",", ".")
    try:
        return float(t)
    except Exception:
        return None

def parse_int_or_none(txt: str):
    t = (txt or "").strip()
    if t == "" or t.lower() in ("na","none","null","-"):
        return None
    try:
        return int(t)
    except Exception:
        return None

def valid_date(s: str):
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except Exception:
        return False

# ==================== /listar_hoy ====================

async def listar_hoy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        ensure_csv(UPCOMING_CSV, UPCOMING_COLS)
        if not os.path.exists(UPCOMING_CSV):
            await update.message.reply_text("No hay upcoming.csv aún.")
            return
        df = pd.read_csv(UPCOMING_CSV, parse_dates=["date"])
        hoy = datetime.now(TZ).date()
        hoy_df = df[df["date"].dt.date == hoy]
        if hoy_df.empty:
            await update.message.reply_text("Hoy no hay partidos cargados en upcoming.csv.")
            return
        lines = ["*Partidos de HOY en upcoming.csv:*"]
        for _, r in hoy_df.iterrows():
            lines.append(
                f"• {r.get('league','')} — {r['home']} vs {r['away']} "
                f"(1:{r.get('home_odds','')} X:{r.get('draw_odds','')} 2:{r.get('away_odds','')} | "
                f"O2.5:{r.get('ou25_over_odds','')} | C>9.5:{r.get('corners_over95_odds','')} | T>4.5:{r.get('cards_over45_odds','')} | "
                f"DO 1X:{r.get('dc_1x_odds','')} X2:{r.get('dc_x2_odds','')} 12:{r.get('dc_12_odds','')})"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text("Error listando hoy.")

# ==================== Flujo /cargar (upcoming.csv) ====================

(
    D_FECHA, D_LIGA, D_HOME, D_AWAY,
    D_HOME_ODDS, D_DRAW_ODDS, D_AWAY_ODDS,
    D_OU_OVER, D_OU_UNDER, D_COR_ODDS, D_CARD_ODDS,
    D_DC1X, D_DCX2, D_DC12, D_CONFIRM
) = range(15)

async def cargar_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_csv(UPCOMING_CSV, UPCOMING_COLS)
    context.user_data["nuevo_partido"] = {}
    await update.message.reply_text(
        "Vamos a crear un partido nuevo.\n"
        "📅 *Fecha* (YYYY-MM-DD). Ej: 2025-10-03",
        parse_mode=ParseMode.MARKDOWN
    )
    return D_FECHA

async def cargar_fecha(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = (update.message.text or "").strip()
    if not valid_date(s):
        await update.message.reply_text("Formato inválido. Usa YYYY-MM-DD. Ej: 2025-10-03")
        return D_FECHA
    context.user_data["nuevo_partido"]["date"] = s
    await update.message.reply_text("🏆 *Liga/Competición* (Ej: Premier League, La Liga, MLS)", parse_mode=ParseMode.MARKDOWN)
    return D_LIGA

async def cargar_liga(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["league"] = (update.message.text or "").strip()
    await update.message.reply_text("🏠 *Equipo local* (home)")
    return D_HOME

async def cargar_home(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["home"] = (update.message.text or "").strip()
    await update.message.reply_text("🚗 *Equipo visitante* (away)")
    return D_AWAY

async def cargar_away(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["away"] = (update.message.text or "").strip()
    await update.message.reply_text("💰 *Cuota local* (decimal, opcional; deja vacío si no tienes)")
    return D_HOME_ODDS

async def cargar_home_odds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["home_odds"] = parse_float_or_none(update.message.text)
    await update.message.reply_text("💰 *Cuota empate* (decimal, opcional)")
    return D_DRAW_ODDS

async def cargar_draw_odds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["draw_odds"] = parse_float_or_none(update.message.text)
    await update.message.reply_text("💰 *Cuota visitante* (decimal, opcional)")
    return D_AWAY_ODDS

async def cargar_away_odds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["away_odds"] = parse_float_or_none(update.message.text)
    await update.message.reply_text("⚽ *Over 2.5 odds* (decimal, opcional)")
    return D_OU_OVER

async def cargar_ou_over(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["ou25_over_odds"] = parse_float_or_none(update.message.text)
    await update.message.reply_text("⚽ *Under 2.5 odds* (decimal, opcional)")
    return D_OU_UNDER

async def cargar_ou_under(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["ou25_under_odds"] = parse_float_or_none(update.message.text)
    await update.message.reply_text("🚩 *Corners > 9.5 odds* (decimal, opcional)")
    return D_COR_ODDS

async def cargar_cor_odds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["corners_over95_odds"] = parse_float_or_none(update.message.text)
    await update.message.reply_text("🟥 *Tarjetas > 4.5 odds* (decimal, opcional)")
    return D_CARD_ODDS

async def cargar_card_odds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["cards_over45_odds"] = parse_float_or_none(update.message.text)
    await update.message.reply_text("♻️ *Doble Oportunidad 1X* (decimal, opcional)")
    return D_DC1X

async def cargar_dc1x(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["dc_1x_odds"] = parse_float_or_none(update.message.text)
    await update.message.reply_text("♻️ *Doble Oportunidad X2* (decimal, opcional)")
    return D_DCX2

async def cargar_dcx2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["dc_x2_odds"] = parse_float_or_none(update.message.text)
    await update.message.reply_text("♻️ *Doble Oportunidad 12* (decimal, opcional)")
    return D_DC12

async def cargar_dc12(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_partido"]["dc_12_odds"] = parse_float_or_none(update.message.text)

    r = context.user_data["nuevo_partido"]
    resumen = (
        f"Confirma el registro:\n\n"
        f"📅 {r.get('date')}\n"
        f"🏆 {r.get('league')}\n"
        f"🏠 {r.get('home')}  vs  🚗 {r.get('away')}\n"
        f"💰 1X2 ⇒ 1:{r.get('home_odds')} X:{r.get('draw_odds')} 2:{r.get('away_odds')}\n"
        f"⚽ O/U 2.5 ⇒ Over:{r.get('ou25_over_odds')} | Under:{r.get('ou25_under_odds')}\n"
        f"🚩 Corners>9.5 ⇒ {r.get('corners_over95_odds')} | 🟥 Tarjetas>4.5 ⇒ {r.get('cards_over45_odds')}\n"
        f"♻️ DobleOport ⇒ 1X:{r.get('dc_1x_odds')} | X2:{r.get('dc_x2_odds')} | 12:{r.get('dc_12_odds')}\n\n"
        f"Escribe *guardar* para confirmar o *cancelar* para abortar."
    )
    await update.message.reply_text(resumen, parse_mode=ParseMode.MARKDOWN)
    return D_CONFIRM

async def cargar_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip().lower()
    if txt not in ("guardar", "g", "si", "sí"):
        await update.message.reply_text("Cancelado. No se guardó nada.")
        return ConversationHandler.END

    ensure_csv(UPCOMING_CSV, UPCOMING_COLS)
    r = context.user_data.get("nuevo_partido", {})
    oblig = ["date","league","home","away"]
    faltan = [k for k in oblig if not r.get(k)]
    if faltan:
        await update.message.reply_text(f"Faltan campos obligatorios: {', '.join(faltan)}. Se cancela.")
        return ConversationHandler.END

    row = {c: ("" if r.get(c) is None else r.get(c)) for c in UPCOMING_COLS}
    if os.path.exists(UPCOMING_CSV):
        df = pd.read_csv(UPCOMING_CSV)
        for c in UPCOMING_COLS:
            if c not in df.columns:
                df[c] = ""
    else:
        df = pd.DataFrame(columns=UPCOMING_COLS)

    mask = (df["date"] == row["date"]) & (df["home"] == row["home"]) & (df["away"] == row["away"])
    if mask.any():
        df.loc[mask, UPCOMING_COLS] = [row[c] for c in UPCOMING_COLS]
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(UPCOMING_CSV, index=False)

    await update.message.reply_text("✅ Partido guardado en data/upcoming.csv.\nAhora puedes pedir: *apuestas para hoy*.", parse_mode=ParseMode.MARKDOWN)
    return ConversationHandler.END

async def cargar_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Operación cancelada.")
    return ConversationHandler.END

# ==================== Flujo /resultado extendido (matches_hist.csv) ====================

(
    R_FECHA, R_LIGA, R_HOME, R_AWAY,
    R_GH, R_GA, R_CH, R_CA,
    R_SH, R_SA, R_SO_H, R_SO_A,
    R_CDH, R_CDA, R_POSH, R_POSA,
    R_FH, R_FA, R_WEATHER, R_CONFIRM
) = range(100, 120)

async def resultado_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_csv(HIST_CSV, HIST_COLS)
    context.user_data["nuevo_resultado"] = {}
    await update.message.reply_text(
        "Vamos a registrar un *resultado con estadísticas*.\n"
        "📅 Fecha (YYYY-MM-DD)",
        parse_mode=ParseMode.MARKDOWN
    )
    return R_FECHA

async def resultado_fecha(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = (update.message.text or "").strip()
    if not valid_date(s):
        await update.message.reply_text("Formato inválido. Usa YYYY-MM-DD.")
        return R_FECHA
    context.user_data["nuevo_resultado"]["date"] = s
    await update.message.reply_text("🏆 Liga/Competición")
    return R_LIGA

async def resultado_liga(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["league"] = (update.message.text or "").strip()
    await update.message.reply_text("🏠 Equipo local (home)")
    return R_HOME

async def resultado_home(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["home"] = (update.message.text or "").strip()
    await update.message.reply_text("🚗 Equipo visitante (away)")
    return R_AWAY

async def resultado_away(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["away"] = (update.message.text or "").strip()
    await update.message.reply_text("🔢 Goles local (entero)")
    return R_GH

async def resultado_gh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = parse_int_or_none(update.message.text)
    if v is None:
        await update.message.reply_text("Debe ser un número (ej: 2).")
        return R_GH
    context.user_data["nuevo_resultado"]["home_goals"] = v
    await update.message.reply_text("🔢 Goles visitante (entero)")
    return R_GA

async def resultado_ga(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = parse_int_or_none(update.message.text)
    if v is None:
        await update.message.reply_text("Debe ser un número (ej: 1).")
        return R_GA
    context.user_data["nuevo_resultado"]["away_goals"] = v
    await update.message.reply_text("🚩 Corners local (entero, opcional; puedes dejar vacío)")
    return R_CH

async def resultado_ch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = parse_int_or_none(update.message.text)
    context.user_data["nuevo_resultado"]["home_corners"] = v
    await update.message.reply_text("🚩 Corners visitante (entero, opcional; puedes dejar vacío)")
    return R_CA

async def resultado_ca(update: Update, context: ContextTypes.DEFAULT_TYPE):
    v = parse_int_or_none(update.message.text)
    context.user_data["nuevo_resultado"]["away_corners"] = v
    await update.message.reply_text("🎯 Tiros totales LOCAL (entero, opcional)")
    return R_SH

async def resultado_sh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["home_shots"] = parse_int_or_none(update.message.text)
    await update.message.reply_text("🎯 Tiros totales VISITANTE (entero, opcional)")
    return R_SA

async def resultado_sa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["away_shots"] = parse_int_or_none(update.message.text)
    await update.message.reply_text("🎯 Tiros al arco LOCAL (entero, opcional)")
    return R_SO_H

async def resultado_so_h(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["home_shots_on"] = parse_int_or_none(update.message.text)
    await update.message.reply_text("🎯 Tiros al arco VISITANTE (entero, opcional)")
    return R_SO_A

async def resultado_so_a(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["away_shots_on"] = parse_int_or_none(update.message.text)
    await update.message.reply_text("🟥 Tarjetas LOCAL (entero, opcional)")
    return R_CDH

async def resultado_cdh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["home_cards"] = parse_int_or_none(update.message.text)
    await update.message.reply_text("🟥 Tarjetas VISITANTE (entero, opcional)")
    return R_CDA

async def resultado_cda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["away_cards"] = parse_int_or_none(update.message.text)
    await update.message.reply_text("📊 Posesión LOCAL (%) (entero 0–100, opcional)")
    return R_POSH

async def resultado_posh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["home_possession"] = parse_int_or_none(update.message.text)
    await update.message.reply_text("📊 Posesión VISITANTE (%) (entero 0–100, opcional)")
    return R_POSA

async def resultado_posa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["away_possession"] = parse_int_or_none(update.message.text)
    await update.message.reply_text("🚨 Faltas LOCAL (entero, opcional)")
    return R_FH

async def resultado_fh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["home_fouls"] = parse_int_or_none(update.message.text)
    await update.message.reply_text("🚨 Faltas VISITANTE (entero, opcional)")
    return R_FA

async def resultado_fa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["away_fouls"] = parse_int_or_none(update.message.text)
    await update.message.reply_text("🌦️ Clima (clear/rain/storm/snow, opcional)")
    return R_WEATHER

async def resultado_weather(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["nuevo_resultado"]["weather"] = (update.message.text or "").strip() or None
    r = context.user_data["nuevo_resultado"]
    resumen = (
        f"Confirma el resultado:\n\n"
        f"📅 {r.get('date')} | 🏆 {r.get('league')}\n"
        f"🏠 {r.get('home')} {r.get('home_goals')} - {r.get('away_goals')} {r.get('away')}\n"
        f"🚩 Corners: {r.get('home_corners')} - {r.get('away_corners')}\n"
        f"🎯 Tiros: {r.get('home_shots')} - {r.get('away_shots')} | A puerta: {r.get('home_shots_on')} - {r.get('away_shots_on')}\n"
        f"🟥 Tarjetas: {r.get('home_cards')} - {r.get('away_cards')}\n"
        f"📊 Posesión: {r.get('home_possession')}% - {r.get('away_possession')}%\n"
        f"🚨 Faltas: {r.get('home_fouls')} - {r.get('away_fouls')}\n"
        f"🌦️ Clima: {r.get('weather')}\n\n"
        f"Escribe *guardar* para confirmar o *cancelar* para abortar."
    )
    await update.message.reply_text(resumen)
    return R_CONFIRM

async def resultado_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip().lower()
    if txt not in ("guardar", "g", "si", "sí"):
        await update.message.reply_text("Cancelado. No se guardó nada.")
        return ConversationHandler.END

    ensure_csv(HIST_CSV, HIST_COLS)
    r = context.user_data.get("nuevo_resultado", {})
    oblig = ["date","league","home","away","home_goals","away_goals"]
    faltan = [k for k in oblig if (r.get(k) is None or r.get(k) == "")]
    if faltan:
        await update.message.reply_text(f"Faltan campos obligatorios: {', '.join(faltan)}. Se cancela.")
        return ConversationHandler.END

    row = {c: ("" if r.get(c) is None else r.get(c)) for c in HIST_COLS}
    if os.path.exists(HIST_CSV):
        df = pd.read_csv(HIST_CSV)
        for c in HIST_COLS:
            if c not in df.columns:
                df[c] = ""
    else:
        df = pd.DataFrame(columns=HIST_COLS)

    mask = (df["date"] == row["date"]) & (df["home"] == row["home"]) & (df["away"] == row["away"])
    if mask.any():
        df.loc[mask, HIST_COLS] = [row[c] for c in HIST_COLS]
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HIST_CSV, index=False)

    await update.message.reply_text("✅ Resultado guardado en data/matches_hist.csv.")
    return ConversationHandler.END

async def resultado_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Operación cancelada.")
    return ConversationHandler.END

# ==================== Comandos básicos y análisis ====================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

def format_reply(date_str: str, results: list) -> str:
    if not results:
        return f"No encontré partidos para {date_str}."
    lines = [f"*Estudio de partidos para {date_str}* (America/Bogota)"]
    for r in results[:20]:
        lines.append(
            f"\n*{r.get('league','')}*: {r['match']}\n"
            f"• 1X2: Local *{r['p_home']}*, Empate *{r['p_draw']}*, Visita *{r['p_away']}*\n"
            f"• Doble Oportunidad: *{r['double_chance_best']['market']}* (p≈{r['double_chance_best']['p']})\n"
            f"• Goles: Over2.5 p≈{r['p_over25']} | Corners>9.5 p≈{r['p_corners_over95']} | Tarjetas>4.5 p≈{r.get('p_cards_over45','—')} | Upset p≈{r['p_upset']}\n"
        )
        vb = r.get("value_bets") or []
        if vb:
            vb_txt = " | ".join([f"{m['market']} EV={m['ev']:+.3f} Kelly={m['kelly']:.2f}" for m in vb])
            lines.append(f"• *Value bets:* {vb_txt}")
    if len(results) > 20:
        lines.append(f"\n… y {len(results)-20} partidos más.")
    return "\n".join(lines)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        q = update.message.text or ""
        cfg = load_config()
        date_str, res = analyze_today(q, cfg, hist_csv=HIST_CSV, up_csv=UPCOMING_CSV)
        reply = format_reply(date_str, res)
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        logging.exception("Error en on_text")
        await update.message.reply_text("Hubo un problema consultando datos. Intento con CSVs locales…")
        date_str, res = analyze_today(update.message.text or "", {}, hist_csv=HIST_CSV, up_csv=UPCOMING_CSV)
        await update.message.reply_text(format_reply(date_str, res), parse_mode=ParseMode.MARKDOWN)

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.exception("Excepción no manejada", exc_info=context.error)

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Falta TELEGRAM_BOT_TOKEN")

    logging.basicConfig(level=logging.INFO)
    external = os.environ.get("WEBHOOK_URL") or os.environ.get("RENDER_EXTERNAL_URL")
    if not external:
        raise SystemExit("Falta WEBHOOK_URL o RENDER_EXTERNAL_URL")

    async def post_init(app: Application):
        url = f"{external.rstrip('/')}/webhook"
        await app.bot.set_webhook(url, drop_pending_updates=True)
        info = await app.bot.get_webhook_info()
        logging.info("Webhook configurado: %s | pendientes=%s", info.url, info.pending_update_count)

    app = Application.builder().token(token).post_init(post_init).build()
    app.add_error_handler(on_error)

    # Comandos
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("listar_hoy", listar_hoy))

    # Conversations: /cargar y /resultado extendido
    conv_cargar = ConversationHandler(
        entry_points=[CommandHandler("cargar", cargar_start)],
        states={
            D_FECHA: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_fecha)],
            D_LIGA: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_liga)],
            D_HOME: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_home)],
            D_AWAY: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_away)],
            D_HOME_ODDS: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_home_odds)],
            D_DRAW_ODDS: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_draw_odds)],
            D_AWAY_ODDS: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_away_odds)],
            D_OU_OVER: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_ou_over)],
            D_OU_UNDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_ou_under)],
            D_COR_ODDS: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_cor_odds)],
            D_CARD_ODDS: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_card_odds)],
            D_DC1X: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_dc1x)],
            D_DCX2: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_dcx2)],
            D_DC12: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_dc12)],
            D_CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, cargar_confirm)],
        },
        fallbacks=[CommandHandler("cancelar", cargar_cancel)],
        allow_reentry=True,
    )
    app.add_handler(conv_cargar)

    conv_resultado = ConversationHandler(
        entry_points=[CommandHandler("resultado", resultado_start)],
        states={
            R_FECHA: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_fecha)],
            R_LIGA: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_liga)],
            R_HOME: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_home)],
            R_AWAY: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_away)],
            R_GH: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_gh)],
            R_GA: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_ga)],
            R_CH: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_ch)],
            R_CA: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_ca)],
            R_SH: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_sh)],
            R_SA: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_sa)],
            R_SO_H: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_so_h)],
            R_SO_A: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_so_a)],
            R_CDH: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_cdh)],
            R_CDA: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_cda)],
            R_POSH: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_posh)],
            R_POSA: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_posa)],
            R_FH: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_fh)],
            R_FA: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_fa)],
            R_WEATHER: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_weather)],
            R_CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, resultado_confirm)],
        },
        fallbacks=[CommandHandler("cancelar", resultado_cancel)],
        allow_reentry=True,
    )
    app.add_handler(conv_resultado)

    # Handler general de texto (consultas de apuestas)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    port = int(os.environ.get("PORT", "10000"))
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path="webhook",
        webhook_url=f"{external.rstrip('/')}/webhook",
        drop_pending_updates=True,
    )

if __name__ == "__main__":
    main()
