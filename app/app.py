from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os

model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "stock_model.pkl")
model = joblib.load(model_path)

def create_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df = df.dropna()
    return df

def prepare_data_for_ticker(ticker):
    # 1. Download data
    df = yf.download(ticker, period="5y")
    # If Yahoo returns empty data
    if df is None or df.empty:
        raise ValueError("No data found for this ticker. Check the symbol or try a different one.")
    
    # 2. Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Make sure required columns exist
    required_cols = ["Close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError("Downloaded data missing required columns.")
    
    # 3. Copy the df before modifying
    df = df.copy()

    # 4. Feature engineering
    df['Return'] = df['Close'].pct_change()
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()

    # 5. Drop rows with NaNs in feature columns
    features = ['Lag1', 'Lag2', 'MA5', 'MA10', 'Return']
    df = df.dropna(subset=features)

    # If after cleaning, df is empty
    if df.empty:
        raise ValueError("Not enough data to compute required features (MA10, Lag values). Try a stock with longer history.")

    # 6. Take last valid row
    latest = df[features].tail(1)

    # Final check
    if latest.shape[0] == 0:
        raise ValueError("Failed to compute latest features. Try another ticker.")

    return latest



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        ticker = request.form["ticker"].upper()

        try:
            latest = prepare_data_for_ticker(ticker)
            
            pred_value = model.predict(latest)[0]
            prediction = f"Predicted Next-Day Closing Price for {ticker}: ${pred_value:.2f}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
