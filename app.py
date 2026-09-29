import time
import requests
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
# 1. FINBERT SENTIMENT ANALYSER (SNEL & VEILIG)
# ==========================================
class FinBertSentiment:
    def __init__(self):
        print("⏳ FinBERT model wordt geladen (dit kan de allereerste keer 1-2 minuten duren)...")
        start_time = time.time()
        
        # Laad tokenizer en model
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.labels = ["positive", "negative", "neutral"]
        
        print(f"✅ FinBERT succesvol geladen in {time.time() - start_time:.1f} seconden!\n")

    def analyze_news(self, headlines):
        if not headlines:
            return 0.0

        inputs = self.tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        pos_score = predictions[:, 0].mean().item()
        neg_score = predictions[:, 1].mean().item()
        return pos_score - neg_score

def get_latest_news_sentiment(ticker, finbert_model):
    """
    Haalt nieuwskoppen op met een snelle fallback om vasthangen te voorkomen.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        # Soms blijft yf_ticker.news hangen; we pakken maximaal 5 artikelen
        news = yf_ticker.news
        headlines = []
        
        if news:
            for item in news[:5]:
                # Yahoo Finance format ondersteuning
                title = item.get('title') or item.get('content', {}).get('title')
                if title:
                    headlines.append(title)
        
        if headlines:
            print(f"📰 Laatste nieuwskoppen voor {ticker}:")
            for h in headlines[:3]:
                print(f" - {h}")
            return finbert_model.analyze_news(headlines)
        else:
            print(f"ℹ️ Geen recent nieuws gevonden voor {ticker}. Sentiment ingesteld op neutraal (0.0).")
            return 0.0
            
    except Exception as e:
        print(f"⚠️ Kon nieuws niet ophalen ({e}). We gaan door met alleen technische analyse.")
        return 0.0
