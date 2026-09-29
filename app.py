import sys
import time
import os

# Schakel HuggingFace parallelisme waarschuwingen en lussen uit
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import lightgbm as lgb

# STEL HIER IN: Wil je FinBERT gebruiken? (True = met nieuws sentiment, False = ultrasnel alleen technisch)
USE_FINBERT = True

if USE_FINBERT:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. FINBERT SENTIMENT CLASS
# ==========================================
class FinBertSentiment:
    def __init__(self):
        print("[1/4] ⏳ FinBERT AI Sentiment model laden van HuggingFace...")
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        print(f"[1/4] ✅ FinBERT geladen in {time.time() - start_time:.1f} seconden.\n")

    def analyze_news(self, headlines):
        if not headlines:
            return 0.0
        inputs = self.tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pos_score = predictions[:, 0].mean().item()
        neg_score = predictions[:, 1].mean().item()
        return pos_score - neg_score

# ==========================================
# 2. DATA OPHALEN & FEATURE ENGINEERING
# ==========================================
def fetch_and_prepare_data(ticker, period="3y"):
    print(f"   --> Koersdata ophalen voor {ticker} via Yahoo Finance...")
    
    # threads=False voorkomt dat yfinance blijft hangen in een oneindige lus
    df = yf.download(ticker, period=period, interval="1d", progress=False, threads=False)
    
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1)

    # Indicatoren berekenen
    df['EMA_5'] = ta.trend.ema_indicator(df['Close'], window=5)
    df['EMA_15'] = ta.trend.ema_indicator(df['Close'], window=15)
    df['EMA_Diff'] = (df['EMA_5'] - df['EMA_15']) / df['Close']
    
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
    
    macd = ta.trend.MACD(df['Close'])
    df['MACD_Hist'] = macd.macd_diff()
    
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Vol_SMA20']

    # Target (3-5 dagen horizon)
    horizon = 5
    profit_target = 0.03
    stop_loss = 0.015
    
    targets = []
    for i in range(len(df) - horizon):
        future_prices = df['Close'].iloc[i+1 : i+1+horizon]
        entry_price = df['Close'].iloc[i]
        
        max_return = (future_prices.max() - entry_price) / entry_price
        min_return = (future_prices.min() - entry_price) / entry_price
        
        if max_return >= profit_target and min_return > -stop_loss:
            targets.append(1)
        else:
            targets.append(0)
            
    targets.extend([np.nan] * horizon)
    df['Target'] = targets
    df.dropna(inplace=True)
    return df

# ==========================================
# 3. AI MODEL TRAINING
# ==========================================
def train_swing_model(df):
    print("   --> AI Model (LightGBM) trainen op historische patroongegevens...")
    features = ['EMA_Diff', 'RSI', 'Stoch_K', 'MACD_Hist', 'Volume_Ratio', 'ATR']
    
    X = df[features]
    y = df['Target']
    
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=4,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model, features

# ==========================================
# 4. NIEUWS SENTIMENT OPHALEN
# ==========================================
def get_news_sentiment(ticker, finbert_model):
    if not USE_FINBERT or finbert_model is None:
        return 0.0
    
    print(f"   --> Nieuws ophalen en FinBERT analyse uitvoeren voor {ticker}...")
    try:
        yf_ticker = yf.Ticker(ticker)
        news = yf_ticker.news
        headlines = []
        
        if news:
            for item in news[:5]:
                title = item.get('title') or item.get('content', {}).get('title')
                if title:
                    headlines.append(title)
        
        if headlines:
            print("   📰 Laatste Nieuwskoppen:")
            for h in headlines[:2]:
                print(f"      - {h}")
            return finbert_model.analyze_news(headlines)
        else:
            print("   ℹ️ Geen recent nieuws gevonden, sentiment op neutraal (0.0).")
            return 0.0
    except Exception as e:
        print(f"   ⚠️ Fout bij ophalen nieuws ({e}). Sentiment genegeerd.")
        return 0.0

# ==========================================
# 5. HOOFDFUNCTIE VOOR ANALYSE
# ==========================================
def analyze_stock(ticker, finbert_model):
    print(f"\n==========================================")
    print(f" 🔍 START ANALYSE: {ticker}")
    print(f"==========================================")
    
    df = fetch_and_prepare_data(ticker)
    if df is None or len(df) < 50:
        print(f"❌ Geen of onvoldoende data gevonden voor ticker '{ticker}'.")
        return

    ai_model, features = train_swing_model(df)
    
    # 1. Technische AI Score
    latest_data = df[features].iloc[-1:]
    latest_close = float(df['Close'].iloc[-1])
    latest_atr = float(df['ATR'].iloc[-1])
    prob_tech = float(ai_model.predict_proba(latest_data)[0][1])

    # 2. Sentiment Score
    sentiment_score = get_news_sentiment(ticker, finbert_model)
    sentiment_prob = (sentiment_score + 1) / 2  # Omzetten naar 0.0 - 1.0 schaal

    # 3. Gewogen Score
    if USE_FINBERT:
        ensemble_score = (prob_tech * 0.70) + (sentiment_prob * 0.30)
    else:
        ensemble_score = prob_tech

    # Resultaten tonen
    print(f"\n--- 📊 EINDRESULTAAT VOOR {ticker} ---")
    print(f"• Huidige Koers      : ${latest_close:.2f}")
    print(f"• Technische AI Score : {prob_tech * 100:.1f}%")
    if USE_FINBERT:
        print(f"• FinBERT Sentiment   : {sentiment_score:+.2f} (Schaal -1.0 tot +1.0)")
    print(f"• TOTAAL AI SCORE     : {ensemble_score * 100:.1f}%")

    if ensemble_score >= 0.60:
        entry_price = latest_close
        stop_loss = entry_price - (1.5 * latest_atr)
        take_profit = entry_price + (3.0 * latest_atr)
        
        print("\n🚀 SIGNAAL: BUY / LONG SETUP (3-5 Dagen) 🚀")
        print(f"• Entry Zone  : ${entry_price:.2f}")
        print(f"• Stop-Loss   : ${stop_loss:.2f}")
        print(f"• Take-Profit : ${take_profit:.2f}")
        print(f"• Risk/Reward : 1:2.0")
    else:
        print("\n⏳ SIGNAAL: GEEN SETUP / WATCHLIST ⏳")
        print("Score is onder de 60%. Wacht op een betere instapkans.")

# ==========================================
# CRUCIALE WRAPPER VOOR WINDOWS/MAC
# ==========================================
if __name__ == '__main__':
    print("=== START AI SWINGTRADE BOT ===")
    
    finbert = None
    if USE_FINBERT:
        finbert = FinBertSentiment()
    
    while True:
        try:
            ticker_input = input("\nVoer US Ticker in (bijv. NVDA, AAPL, TSLA) of 'STOP': ").strip().upper()
            if ticker_input == 'STOP' or ticker_input == '':
                print("Programma gestopt.")
                break
            
            analyze_stock(ticker_input, finbert)
            
        except KeyboardInterrupt:
            print("\nAfgebroken door gebruiker.")
            break
