# ---------------------------------------------------------
#                   🔐 LOGIN SYSTEM
# ---------------------------------------------------------

import streamlit as st

USERNAME = "admin"
PASSWORD = "1234"

# Create session state variable
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Login ")
    st.write("Please enter your login credentials.")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login Successful! Loading the app...")
            st.rerun()
        else:
            st.error("Incorrect username or password")

# If not logged in, show login page and stop execution
if not st.session_state.logged_in:
    login_page()
    st.stop()



import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# ---- Streamlit UI ----
st.set_page_config(layout="wide")
st.title("Stock Price Prediction ")

# Sidebar for user inputs
st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
stock = st.sidebar.text_input("Enter Stock Ticker (e.g., GOOG)", "GOOG").strip()
epochs = st.sidebar.slider("Number of Epochs", 1, 50, 2)
batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# ---- Dates ----
end = dt.datetime.now()
start = dt.datetime(end.year - 20, end.month, end.day)

st.write(f"Fetching data for **{stock}** from {start.date()} to {end.date()} ...")

@st.cache_data(show_spinner=False)
def fetch_data(sym, start, end):
    # Try 1: download (auto_adjust provides adjusted OHLC; 'Adj Close' often absent)
    df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # If empty or missing any 'close' column, try Ticker.history
    if df.empty or not any("close" in str(c).lower() for c in df.columns):
        tkr = yf.Ticker(sym)
        df2 = tkr.history(start=start, end=end, auto_adjust=True)
        if isinstance(df2.columns, pd.MultiIndex):
            df2.columns = df2.columns.get_level_values(-1)
        # Prefer the non-empty one
        if (not df2.empty) and (len(df2.columns) >= len(df.columns)):
            df = df2

    # Last resort: rename any variant of close to 'Adj Close'
    close_col = None
    for c in df.columns:
        if "close" in str(c).lower():
            close_col = c
            break

    if close_col is not None:
        # Ensure an 'Adj Close' exists for downstream code
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df[close_col]
    return df

stock_data = fetch_data(stock, start, end)

# ---- Diagnostics & Guards ----
if stock_data.empty:
    st.error(f"No data returned for `{stock}`. Try a different ticker (AAPL, TCS.NS, RELIANCE.NS) or check internet.")
    st.stop()

if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(-1)

# Create Adj Close if still missing and any 'close' variant exists
if "Adj Close" not in stock_data.columns:
    # try to find any column with 'close' in its name
    candidates = [c for c in stock_data.columns if "close" in str(c).lower()]
    if candidates:
        stock_data["Adj Close"] = stock_data[candidates[0]]

# Final guard
if "Adj Close" not in stock_data.columns or stock_data["Adj Close"].isna().all():
    st.error(
        "Price columns missing or empty for this ticker.\n\n"
        "Returned columns: " + ", ".join(map(str, stock_data.columns))
    )
    st.info("Tip: Try `AAPL`, `GOOG`, `TCS.NS`, `RELIANCE.NS`, `HDFCBANK.NS`.")
    st.stop()

# ---- Derived columns (for charts) ----
stock_data["MA_100"] = stock_data["Adj Close"].rolling(100).mean()
stock_data["MA_250"] = stock_data["Adj Close"].rolling(250).mean()
stock_data["percent_change"] = stock_data["Adj Close"].pct_change()

# ---- Plot: Adjusted Closing Price ----
st.subheader("Adjusted Closing Price")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data["Adj Close"], label="Adjusted Close Price")
ax.set_xlabel("Years")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# ---- Plot: 100-Day Moving Average ----
st.subheader("100-Day Moving Average")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data["Adj Close"], alpha=0.5, label="Actual Data", color="gray")
ax.plot(stock_data["MA_100"], label="100-Day MA", linestyle="dashed")
ax.set_xlabel("Years")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# ---- Plot: 250-Day Moving Average ----
st.subheader("250-Day Moving Average")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data["Adj Close"], alpha=0.5, label="Actual Data", color="gray")
ax.plot(stock_data["MA_250"], label="250-Day MA", linestyle="dotted")
ax.set_xlabel("Years")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# ---- Plot: Percentage Change ----
st.subheader("Percentage Change in Price")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data["percent_change"], label="Percentage Change", linestyle="solid")
ax.set_xlabel("Years")
ax.set_ylabel("Change")
ax.legend()
st.pyplot(fig)

# ---- Preprocessing for LSTM ----
adj_close = stock_data[["Adj Close"]].dropna()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(adj_close)

# Need at least 101 rows to form one training sample
if len(scaled_data) < 101:
    st.error("Not enough data to train (need at least 101 days). Try a broader date range or a different ticker.")
    st.stop()

X_data, y_data = [], []
for i in range(100, len(scaled_data)):
    X_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

X_data, y_data = np.array(X_data), np.array(y_data)
split = int(len(X_data) * 0.7)
X_train, y_train = X_data[:split], y_data[:split]
X_test, y_test = X_data[split:], y_data[split:]

# ---- Model definition ----
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")

# ---- Train & Visualize ----
if st.sidebar.button("Train Model"):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    st.success("Model Training Completed!")

    # Predictions
    predictions = model.predict(X_test)
    inv_preds = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_test)

    # Plot: Test Data Predictions vs Actual
    st.subheader("Test Data Predictions vs Actual Data")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data.index[split + 100:], inv_y_test, label="Actual")
    ax.plot(stock_data.index[split + 100:], inv_preds, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    # Plot: Whole Data Predictions
    st.subheader("Whole Data Predictions")
    combined_df = pd.concat(
        [
            stock_data[["Adj Close"]][:split + 100],
            pd.DataFrame(inv_preds, index=stock_data.index[split + 100:], columns=["Predicted"])
        ],
        axis=0
    )
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data.index, stock_data["Adj Close"], label=f"(Adj Close, {stock})", color="blue")
    ax.plot(stock_data.index[-len(y_test):], inv_y_test, label="original test data", color="orange")
    ax.plot(stock_data.index[-len(y_test):], inv_preds, label="preds", color="green")
    ax.set_xlabel("Years")
    ax.set_ylabel("Whole Data")
    ax.set_title("Whole Data of the Stock")
    ax.legend()
    st.pyplot(fig)

    # Future Predictions (Next 7 days)
    future_days = 7
    last_100_days = scaled_data[-100:]
    X_future = np.array([last_100_days])

    future_preds = []
    current_input = X_future[0].copy()
    for _ in range(future_days):
        pred = model.predict(current_input.reshape(1, 100, 1))
        future_preds.append(pred[0, 0])
        current_input = np.append(current_input[1:], pred, axis=0)

    future_preds = np.array(future_preds).reshape(-1, 1)
    inv_future_preds = scaler.inverse_transform(future_preds)
    future_dates = pd.date_range(start=stock_data.index[-1] + pd.DateOffset(1), periods=future_days)

    st.subheader("Future Stock Price Predictions (Next 7 Days)")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(future_dates, inv_future_preds, marker="o", linestyle="dashed", label="Future Prediction")
    ax.legend()
    st.pyplot(fig)

st.caption("Tip: If a ticker fails, try AAPL / TCS.NS / RELIANCE.NS. Use a stable internet connection for Yahoo Finance.")


