# predict.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

MODEL_PATH = "stock_predictor.pkl"

def compute_technical_indicators(df):
    df = df.copy()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df = df.dropna()
    return df

def prepare_features(merged):
    merged = compute_technical_indicators(merged)
    # Simple sentiment score: positive - negative
    merged['sentiment_score'] = merged.get('positive', 0) - merged.get('negative', 0)
    merged = merged.dropna()
    return merged

def train_predictive_model(merged):
    merged = prepare_features(merged)
    merged['Target'] = (merged['Close'].shift(-1) > merged['Close']).astype(int)
    merged = merged.dropna()

    features = ['sentiment_score', 'SMA_5', 'SMA_10', 'RSI', 'Close']
    X = merged[features]
    y = merged['Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model

def predict_next_movement(merged):
    if not os.path.exists(MODEL_PATH):
        model = train_predictive_model(merged)
    else:
        model = joblib.load(MODEL_PATH)

    merged = prepare_features(merged)
    latest = merged.iloc[[-1]][['sentiment_score', 'SMA_5', 'SMA_10', 'RSI', 'Close']]
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][1]

    return pred, prob
