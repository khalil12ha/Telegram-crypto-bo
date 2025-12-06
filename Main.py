#!/usr/bin/env python3
"""
Telegram Crypto Analyzer Bot
- Scans centralized exchanges via ccxt
- Downloads OHLCV, computes peaks & troughs
- Generates chart images showing highs/lows and entry/exit zones
- Sends messages and images to Telegram
"""

import os
import time
import logging
from io import BytesIO
from datetime import datetime, timezone

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from telegram import Bot
from telegram.error import TelegramError

# ENV VARIABLES
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
SCAN_INTERVAL = int(os.getenv('SCAN_INTERVAL_SECONDS', '300'))
MAX_SYMBOLS_PER_EXCHANGE = int(os.getenv('MAX_SYMBOLS_PER_EXCHANGE', '200'))
OHLCV_LIMIT = int(os.getenv('OHLCV_LIMIT', '200'))
EXCHANGES_WHITELIST = os.getenv('EXCHANGES_WHITELIST', '')  # empty = all ccxt exchanges

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger('crypto-analyzer')

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    logger.error("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID")
    raise SystemExit("Missing TOKEN or CHAT_ID")

bot = Bot(token=TELEGRAM_TOKEN)


# --- RSI indicator ---
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


# --- find peaks & troughs ---
def find_peaks_troughs(df: pd.DataFrame, window: int = 5):
    highs = df['high'].rolling(window=window, center=True).apply(lambda x: 1 if x[window//2] == max(x) else 0)
    lows = df['low'].rolling(window=window, center=True).apply(lambda x: 1 if x[window//2] == min(x) else 0)
    peaks = df[highs == 1]
    troughs = df[lows == 1]
    return peaks, troughs


# --- Plot chart image ---
def plot_chart(df, symbol, exchange_id, peaks, troughs, entry_zones, exit_zones):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close')
    plt.plot(df.index, df['ma50'], label='MA50')
    plt.plot(df.index, df['ma200'], label='MA200')

    plt.scatter(peaks.index, peaks['high'], marker='^', label='Peaks')
    plt.scatter(troughs.index, troughs['low'], marker='v', label='Troughs')

    for z in entry_zones:
        plt.axhspan(z[0], z[1], alpha=0.12, color='green')
    for z in exit_zones:
        plt.axhspan(z[0], z[1], alpha=0.12, color='red')

    plt.title(f"{exchange_id.upper()} — {symbol} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf


# --- Suggest entry/exit zones ---
def suggest_zones(df: pd.DataFrame):
    last_close = df['close'].iloc[-1]
    std = df['close'].pct_change().rolling(20).std().iloc[-1]
    if pd.isna(std) or std == 0:
        std = 0.01

    k = 6
    entry_low = last_close * (1 - k * std)
    entry_high = last_close * (1 - 0.5 * k * std)
    exit_low = last_close * (1 + 0.5 * k * std)
    exit_high = last_close * (1 + k * std)

    return [(entry_low, entry_high)], [(exit_low, exit_high)]


# --- Spot or Futures suggestion ---
def suggest_market_type(df):
    rsi = df['rsi'].iloc[-1]
    ma50 = df['ma50'].iloc[-1]
    ma200 = df['ma200'].iloc[-1]

    if rsi < 35 and ma50 < ma200:
        return "Spot (Buy Zone)"
    if rsi > 65 and ma50 > ma200:
        return "Futures (Momentum)"
    return "Spot (Conservative)"


# --- Analyze one symbol ---
def analyze_symbol(exchange, symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=OHLCV_LIMIT)
        if not ohlcv:
            return None

        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)

        df['ma50'] = df['close'].rolling(50).mean()
        df['ma200'] = df['close'].rolling(200).mean()
        df['rsi'] = compute_rsi(df['close'])

        peaks, troughs = find_peaks_troughs(df, 7)
        entry_zones, exit_zones = suggest_zones(df)
        market_suggestion = suggest_market_type(df)

        last_close = df['close'].iloc[-1]

        text = (
            f"{exchange.id.upper()} — {symbol}\n"
            f"Price: {last_close:.8g}\n"
            f"RSI: {df['rsi'].iloc[-1]:.2f}\n"
            f"MA50: {df['ma50'].iloc[-1]:.8g}  MA200: {df['ma200'].iloc[-1]:.8g}\n"
            f"Suggestion: {market_suggestion}\n"
            f"Entry zone: {entry_zones[0][0]:.8g} — {entry_zones[0][1]:.8g}\n"
            f"Exit zone: {exit_zones[0][0]:.8g} — {exit_zones[0][1]:.8g}\n"
        )

        img = plot_chart(df, symbol, exchange.id, peaks, troughs, entry_zones, exit_zones)
        return {"text": text, "image": img}

    except Exception as e:
        logger.error(f"Error analyzing {symbol} on {exchange.id}: {e}")
        return None


# --- Send to Telegram ---
def send_telegram_report(payload):
    try:
        if payload.get("image"):
            bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=payload["image"],
                caption=payload["text"]
            )
        else:
            bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=payload["text"]
            )
    except TelegramError as te:
        logger.error(f"Telegram error: {te}")


# --- Load CCXT exchanges ---
def build_exchanges(whitelist=""):
    exs = {}
    supported = ccxt.exchanges

    chosen = [e.strip() for e in whitelist.split(',') if e.strip()] if whitelist else supported

    for ex_id in chosen:
        try:
            exchange_cls = getattr(ccxt, ex_id)
            ex = exchange_cls({"enableRateLimit": True})
            if hasattr(ex, "fetch_ohlcv"):
                ex.load_markets()
                exs[ex_id] = ex
                logger.info(f"Loaded exchange: {ex_id} with {len(ex.markets)} markets")
        except Exception:
            logger.warning(f"Skipping exchange {ex_id}")
    return exs


# --- Main loop ---
def main_loop():
    exchanges = build_exchanges(EXCHANGES_WHITELIST)

    while True:
        for ex_id, ex in exchanges.items():
            try:
                symbols = list(ex.markets.keys())[:MAX_SYMBOLS_PER_EXCHANGE]

                for symbol in symbols:
                    if not ("USDT" in symbol or "USD" in symbol or "USDC" in symbol):
                        continue

                    result = analyze_symbol(ex, symbol)
                    if result:
                        send_telegram_report(result)
                        time.sleep(1)  # avoid spamming

            except Exception as e:
                logger.error(f"Error scanning {ex_id}: {e}")

        logger.info(f"Sleeping {SCAN_INTERVAL} seconds...")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main_loop()
