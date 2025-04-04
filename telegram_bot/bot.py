import logging
import requests
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

BOT_TOKEN = "7821507150:AAEstbwV9hWhfz0WiryHps5ix7Bb7lmoRkY"  # <-- ZDE vlož svůj token z @BotFather
API_URL = "https://mlsoccer.onrender.com/predict"

logging.basicConfig(level=logging.INFO)

def handle_message(update: Update, context: CallbackContext):
    text = update.message.text.strip()
    chat_id = update.message.chat_id

    try:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 3:
            update.message.reply_text("❌ Zadej: LIGA,DOMÁCÍ,HOST (např. SP1,Barcelona,Sevilla)")
            return

        league, home, away = parts
        response = requests.post(API_URL, json={
            "league_code": league,
            "home_team": home,
            "away_team": away
        })

        if response.status_code != 200:
            update.message.reply_text(f"⚠️ Chyba serveru: {response.status_code}")
            return

        
        
        data = response.json()
        
        rf_percent = data['rf_prob'] * 100
        xgb_percent = data['xgb_prob'] * 100

        rf_odds = round(1 / data['rf_prob'], 2) if data['rf_prob'] > 0 else "∞"
        xgb_odds = round(1 / data['xgb_prob'], 2) if data['xgb_prob'] > 0 else "∞"
        
        reply = (
            f"📊 *Predikce Over 2.5* pro {data['home_team']} vs. {data['away_team']}:\n\n"
            f"🔹 *Random Forest:* {'✅ ANO' if data['rf_pred'] else '❌ NE'} "
            f"({rf_percent:.2f} % / kurz {rf_odds}) – {data['rf_conf']}\n"
            f"🔹 *XGBoost:* {'✅ ANO' if data['xgb_pred'] else '❌ NE'} "
            f"({xgb_percent:.2f} % / kurz {xgb_odds}) – {data['xgb_conf']}"
        )
        update.message.reply_text(reply, parse_mode="Markdown")
    except Exception as e:
        logging.exception("Chyba při zpracování zprávy:")
        update.message.reply_text(f"⚠️ Došlo k chybě: {e}")

def main():
    updater = Updater(BOT_TOKEN)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
