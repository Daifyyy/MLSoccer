# telegram_bot/bot.py
import requests
import telegram
from telegram.ext import Updater, MessageHandler, Filters

BOT_TOKEN = "TVŮJ_BOT_TOKEN"
API_URL = "http://localhost:8000/predict"

def handle_message(update, context):
    text = update.message.text.strip()
    parts = text.split(',')
    if len(parts) != 3:
        update.message.reply_text("❌ Správný formát: LIGA,DOMÁCÍ,HOST")
        return
    league, home, away = [p.strip() for p in parts]
    try:
        response = requests.post(API_URL, json={
            "league_code": league,
            "home_team": home,
            "away_team": away
        }).json()

        if "error" in response:
            update.message.reply_text(response["error"])
        else:
            msg = (
                f"📊 *Predikce Over 2.5* pro {home} vs. {away}:\n\n"
                f"🔹 *Random Forest:* {response['rf_prob']:.2%} → {'✅ ANO' if response['rf_pred'] else '❌ NE'} ({response['rf_conf']})\n"
                f"🔹 *XGBoost:* {response['xgb_prob']:.2%} → {'✅ ANO' if response['xgb_pred'] else '❌ NE'} ({response['xgb_conf']})"
            )
            update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        update.message.reply_text(f"⚠️ Chyba: {e}")

updater = Updater(BOT_TOKEN, use_context=True)
dp = updater.dispatcher
dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
updater.start_polling()
updater.idle()
