"""
Test Telegram Bot Integration.
Usage:
  python test_telegram.py --token "YOUR_BOT_TOKEN" --chat_id "YOUR_CHAT_ID"
"""

import sys
import argparse
from core.telegram_notifier import send_telegram_message

def test_telegram(token: str, chat_id: str):
    msg = """🚀 *Antigravity Trade Monitor: Telegram Integration Active!* 🚀

Your autonomous agent squad is now connected to this Telegram chat.
Whenever a stock or index triggers an excessive shootup breakout, you will receive real-time alerts right here on your phone, even when your laptop is closed!

📊 *Monitored Markets*:
• US Equities & Indexes (S&P 500, Nasdaq, High Short-Interest Growth)
• Indian Equities & Indexes (NSE NIFTY 50, Bank Nifty, High-Momentum Leaders)

_Status: System Online & Stalking Triggers._"""

    print(f"Sending test alert to chat ID: {chat_id} via @Stocksgannuu_bot...")
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": "Markdown"
    }

    try:
        import requests
        resp = requests.post(url, json=payload, timeout=10)
        data = resp.json()
        if data.get("ok"):
            print("\n[SUCCESS] Test message successfully delivered to your Telegram app!")
            print("Credentials are saved in .env. You are all set to receive automatic trade alerts!")
        else:
            err = data.get("description", "Unknown error")
            print(f"\n[TELEGRAM API NOTICE]: {err}")
            if "chat not found" in err.lower():
                print("\n--> ACTION REQUIRED:")
                print("1. Open Telegram on your phone.")
                print("2. Search for your bot: @Stocksgannuu_bot")
                print("3. Tap 'START' (or send /start).")
                print("4. Re-run this test script!")
    except Exception as e:
        print(f"\n[NETWORK ERROR]: {e}")

if __name__ == "__main__":
    import os
    env_token = None
    env_chat = None
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "TELEGRAM_BOT_TOKEN":
                        env_token = v.strip()
                    elif k.strip() == "TELEGRAM_CHAT_ID":
                        env_chat = v.strip()

    parser = argparse.ArgumentParser(description="Test Telegram Bot Integration")
    parser.add_argument("--token", type=str, default=env_token, help="Telegram Bot Token from @BotFather")
    parser.add_argument("--chat_id", type=str, default=env_chat, help="Your Telegram Chat ID from @userinfobot")
    args = parser.parse_args()

    if not args.token or not args.chat_id:
        print("[ERROR] Please provide --token and --chat_id, or save them in .env")
        sys.exit(1)

    test_telegram(args.token, args.chat_id)
