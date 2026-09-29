import numpy as np
import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import lightgbm as lgb
import ta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. FINBERT SENTIMENT ANALYSER (NLP)
# ==========================================
class FinBertSentiment:
    def __init__(self):
        print("Model FinBERT laden (HuggingFace ProsusAI/finbert)...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.labels = ["positive", "negative", "neutral"]

    def analyze_news(self, headlines):
        """
        Analyseert een lijst met nieuwskoppen en geeft een gewogen Sentiment Score (-1 tot +1).
        """
        if not headlines:
            return 0.0  # Neutraal als er geen nieuws is

        inputs = self.tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Bereken gemiddelde score
        pos_score = predictions[:, 0].mean().item()
        neg_score = predictions[:, 1].mean().item()
        
        # Netto Sentiment Score tussen -1.0 (Zeer Negatief) en +1.0 (Zeer Positief)
        net_sentiment = pos_score - neg_score
        return net_sentiment

def get_latest_news_sentiment(ticker, finbert_model):
    """
    Haalt de laatste nieuwskoppen van Yahoo Finance op voor de ticker.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        news = yf_ticker.news
        headlines = [item['title'] for item in news[:5]] if news else []
        
        if headlines:
            print(f"\n📰 Laatste Nieuwskoppen voor {ticker}:")
            for h in headlines[:3]:
                print(f" - {h}")
            sentiment_score = finbert_model.analyze_news(headlines)
            return sentiment_score
        else:
            return 0.0
    except Exception as e:
        print(f"Kon nieuws niet ophalen: {e}")
        return 0.0

# ==========================================
# 2. DATA PREPARATIE & FEATURE ENGINEERING
# ==========================================
def fetch_and_prepare_data(ticker, period="3y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1)

    # Technical Indicators
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

    # Target (3-5 dagen horizon: +3% Winst vs -1.5% Stop Loss)
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
# 3. AI MODEL TRAINING (LightGBM)
# ==========================================
def train_swing_model(df):
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
# 4. HYBRIDE SWINGTRADE ANALYSER (AI + Sentiment)
# ==========================================
def analyze_stock(ticker, finbert_model):
    print(f"\n==========================================")
    print(f"   ANALISEREN VAN AANDEEL: {ticker}")
    print(f"==========================================")
    
    # Data ophalen
    df = fetch_and_prepare_data(ticker)
    if df is None or len(df) < 100:
        print(f"❌ Ongeldige ticker of te weinig historische data voor {ticker}.")
        return

    # Train model op aandeel-specifieke historie
    ai_model, features = train_swing_model(df)
    
    # 1. Technische AI Score
    latest_data = df[features].iloc[-1:]
    latest_close = float(df['Close'].iloc[-1])
    latest_atr = float(df['ATR'].iloc[-1])
    prob_tech = ai_model.predict_proba(latest_data)[0][1] # Probability (0 tot 1)

    # 2. FinBERT Sentiment Score
    sentiment_score = get_latest_news_sentiment(ticker, finbert_model) # Score tussen -1.0 en +1.0
    sentiment_prob = (sentiment_score + 1) / 2 # Omgezet naar schaal 0 tot 1

    # 3. Hybride Ensemble Score (70% Technisch + 30% FinBERT Sentiment)
    ensemble_score = (prob_tech * 0.70) + (sentiment_prob * 0.30)
    
    # Print Dashboard
    print(f"\n--- 📊 AI SCORECARD FOR {ticker} ---")
    print(f"• Huidige Koers     : ${latest_close:.2f}")
    print(f"• Technische AI Score: {prob_tech * 100:.1f}%")
    print(f"• FinBERT Sentiment  : {sentiment_score:+.2f} (Schaal -1.0 tot +1.0)")
    print(f"• COMBINED AI SCORE  : {ensemble_score * 100:.1f}%")

    # Signaal & Risk Management
    if ensemble_score >= 0.60:
        entry_price = latest_close
        stop_loss = entry_price - (1.5 * latest_atr)
        take_profit = entry_price + (3.0 * latest_atr)
        
        print("\n🚀 CONCLUSION: STRONG BUY / SWING SETUP 🚀")
        print(f"• Signal Status  : BUY / LONG (Horizon: 3-5 Dagen)")
        print(f"• Entry Zone     : ${entry_price:.2f}")
        print(f"• Stop-Loss      : ${stop_loss:.2f} (Risk: ${entry_price - stop_loss:.2f})")
        print(f"• Take-Profit    : ${take_profit:.2f} (Target: ${take_profit - entry_price:.2f})")
        print(f"• Risk/Reward    : 1:2.0")
    else:
        print("\n⏳ CONCLUSION: NO TRADE / WATCHLIST ⏳")
        print("De gecombineerde AI & Sentiment score is te laag (< 60%). Wacht op een beter instapmoment.")

# ==========================================
# MAIN EXECUTION (INTERACTIEVE INVOER)
# ==========================================
if __name__ == "__main__":
    # Laad FinBERT eenmalig op in het geheugen
    finbert = FinBertSentiment()
    
    while True:
        user_input = input("\nVoer een US Ticker in (bijv. NVDA, TSLA, AAPL) of 'STOP' om te sluiten: ").strip().upper()
        if user_input == "STOP" or user_input == "":
            print("Analyse beëindigd.")
            break
        
        analyze_stock(user_input, finbert)
