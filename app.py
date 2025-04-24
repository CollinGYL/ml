# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences, Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("AAPL Stock Price Forecasting")

# -----------------------------------------------------------------------------
# 1) Load Price & Macro Data
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    # Price & Volume
    df_init = yf.download("AAPL", start="2015-01-01", end="2025-01-01", progress=False)
    df_init.columns = df_init.columns.droplevel(1)
    df = df_init[["Close", "Volume"]].copy()
    # Macro: UNRATE, CPI, 1yr/10yr yields
    macro = pdr.DataReader(
        ["UNRATE", "LNS11300060", "CPIAUCSL", "DGS1", "DGS10"],
        "fred",
        df.index.min(),
        df.index.max()
    )
    macro = macro.reindex(df.index).ffill().bfill()
    return df, macro

df, macro = load_data()

# -----------------------------------------------------------------------------
# 2) Fetch News Titles & Dates
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_news():
    ticker     = yf.Ticker("AAPL")
    items      = ticker.news or []
    titles, dates = [], []
    for it in items:
        ts = it.get("providerPublishTime")
        if ts is None: continue
        titles.append(it.get("title",""))
        dates.append(pd.to_datetime(ts, unit="s").date())
    return titles, dates

titles, dates = fetch_news()

# -----------------------------------------------------------------------------
# 3) Train Sentiment Model (IMDB → Embedding+LSTM)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_sent_model():
    num_words, maxlen = 10000, 80
    (x_tr,y_tr),(x_te,y_te) = imdb.load_data(num_words=num_words)
    x_tr = pad_sequences(x_tr, maxlen=maxlen, dtype="int32")
    x_te = pad_sequences(x_te, maxlen=maxlen, dtype="int32")
    m = Sequential([
        Embedding(input_dim=num_words, output_dim=128),
        LSTM(64),
        Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    m.fit(x_tr, y_tr, validation_data=(x_te,y_te), epochs=5, batch_size=128, verbose=2)
    return m, num_words, maxlen

sent_model, num_words, maxlen = train_sent_model()

# -----------------------------------------------------------------------------
# 4) Build Daily Sentiment Series
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def build_sentiment(titles, dates):
    tok = Tokenizer(num_words=num_words)
    tok.fit_on_texts(titles)
    seqs = tok.texts_to_sequences(titles)
    padn = pad_sequences(seqs, maxlen=maxlen, dtype="int32")
    if padn.shape[0] == 0:
        # no news → zeros
        s = pd.Series(0.0, index=df.index, name="daily_sentiment")
    else:
        scores = sent_model.predict(padn, verbose=0).flatten()
        df_s = pd.DataFrame({"date": dates, "sentiment": scores})
        daily = df_s.groupby("date")["sentiment"].mean()
        daily.index = pd.to_datetime(daily.index)
        s = daily.reindex(df.index).ffill().bfill().rename("daily_sentiment")
    return s

daily_sent = build_sentiment(titles, dates)

# -----------------------------------------------------------------------------
# 5) Feature Engineering & Train XGBoost
# -----------------------------------------------------------------------------
def create_features(df_in):
    df_ = df_in.copy()
    idx = df_.index
    df_["dayofweek"]  = idx.dayofweek
    df_["quarter"]    = idx.quarter
    df_["month"]      = idx.month
    df_["year"]       = idx.year
    df_["dayofyear"]  = idx.dayofyear
    df_["dayofmonth"] = idx.day
    df_["weekofyear"] = idx.isocalendar().week
    return df_

def add_lags(df_in):
    df_ = df_in.copy()
    m = df_["Close"].to_dict()
    for lag in [1,2,3]:
        df_[f"lag{lag}"] = (df_.index - pd.Timedelta(f"{lag} days")).map(m)
    return df_

@st.cache_resource
def train_xgb(df, daily_sent, macro):
    feat = create_features(df)
    feat = add_lags(feat)
    feat["daily_sentiment"] = daily_sent
    feat = feat.merge(macro, left_index=True, right_index=True, how="left")
    feat.fillna(method="ffill", inplace=True)
    # Split
    split = "2024-01-01"
    mask  = feat.index < split
    FEATURES = [
        "dayofweek","quarter","month","year",
        "dayofyear","dayofmonth","weekofyear",
        "Volume","lag1","lag2","lag3","daily_sentiment",
        "UNRATE","LNS11300060","CPIAUCSL","DGS1","DGS10"
    ]
    X_tr, y_tr = feat.loc[mask, FEATURES], feat.loc[mask, "Close"]
    X_te, y_te = feat.loc[~mask, FEATURES], feat.loc[~mask, "Close"]
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        early_stopping_rounds=50,
        max_depth=3,
        learning_rate=0.01
    )
    model.fit(X_tr, y_tr, eval_set=[(X_tr,y_tr),(X_te,y_te)], verbose=100)
    y_pr = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pr))
    return feat, mask, y_te, y_pr, rmse

df_feat, mask, y_test, y_pred, rmse = train_xgb(df, daily_sent, macro)

# -----------------------------------------------------------------------------
# 6) Display Results
# -----------------------------------------------------------------------------
st.subheader("Train/Test Split & RMSE")
st.write(f"Test RMSE: {rmse:.2f}")

# Plot actual vs predictions
fig, ax = plt.subplots(figsize=(10,4))
sns.set_style("whitegrid")
df["Close"].plot(ax=ax, label="Actual", color="blue")
ax.scatter(df_feat.index[~mask], y_pred, color="red", s=10, label="Predicted")
ax.axvline("2024-01-01", color="black", linestyle="--")
ax.legend()
st.pyplot(fig)

st.subheader("Monthly Price Distribution")
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.boxplot(data=create_features(df)[["month","Close"]], x="month", y="Close", palette="Blues", ax=ax2)
ax2.set_title("Close Price by Month")
st.pyplot(fig2)

st.subheader("Daily News Sentiment")
st.line_chart(daily_sent)

st.subheader("Macro Indicators Over Time")
st.line_chart(macro)

st.success("Streamlit app loaded successfully!")
