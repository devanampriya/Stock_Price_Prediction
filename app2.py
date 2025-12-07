import streamlit as st

# --------------- SIMPLE LOGIN ---------------

# Change these to whatever you want
VALID_USERNAME = "abcd"
VALID_PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Login")
    st.write("Please login to access the Stock Price Prediction app.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password")

# If not logged in -> show login and stop
if not st.session_state.logged_in:
    login_page()
    st.stop()

# --------------- MAIN APP (your old app2.py) ---------------

import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

st.set_page_config(layout="wide")
st.title("📈 Stock Price Prediction using LSTM")

# Sidebar controls
st.sidebar.header("Controls")
stock = st.sidebar.text_input("Enter Stock Ticker (e.g., GOOG)", "GOOG")
epochs = st.sidebar.slider("Number of Epochs", 1, 50, 2)
batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)

# Date range
end_default = dt.datetime.now().date()
start = st.sidebar.date_input("Start Date", dt.date(end_default.year - 20, end_default.month, end_default.day))
end = st.sidebar.date_input("End Date", end_default)

st.write(f"Fetching data for **{stock}** from **{start}** to **{end}** ...")
stock_data = yf.download(stock, start=start, end=end)

if stock_data.empty:
    st.error("No data found for this ticker / date range. Try AAPL, TCS.NS, RELIANCE.NS, etc.")
    st.stop()

# Decide which price column to use
if "Adj Close" in stock_data.columns:
    price_col = "Adj Close"
elif "Close" in stock_data.columns:
    price_col = "Close"
else:
    st.error("Neither 'Adj Close' nor 'Close' columns found in data.")
    st.stop()

# Moving averages & percent change
stock_data["MA_100"] = stock_data[price_col].rolling(100).mean()
stock_data["MA_250"] = stock_data[price_col].rolling(250).mean()
stock_data["percent_change"] = stock_data[price_col].pct_change()

# Plot: Adjusted Closing Price
st.subheader("Adjusted Closing Price")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data[price_col], label="Adjusted Close Price")
ax.set_xlabel("Years")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Plot: 100-Day MA
st.subheader("100-Day Moving Average")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data[price_col], alpha=0.5, label="Actual Data")
ax.plot(stock_data["MA_100"], linestyle="dashed", label="100-Day MA")
ax.legend()
st.pyplot(fig)

# Plot: 250-Day MA
st.subheader("250-Day Moving Average")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data[price_col], alpha=0.5, label="Actual Data")
ax.plot(stock_data["MA_250"], linestyle="dotted", label="250-Day MA")
ax.legend()
st.pyplot(fig)

# Plot: Percent change
st.subheader("Daily Percentage Change")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data["percent_change"], label="Percent Change")
ax.legend()
st.pyplot(fig)

# ------------ LSTM PREP ------------

data = stock_data[[price_col]]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

X_data, y_data = [], []
for i in range(100, len(scaled_data)):
    X_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

if len(X_data) == 0:
    st.error("Not enough data to train model. Increase the date range.")
    st.stop()

X_data = np.array(X_data)
y_data = np.array(y_data)

split = int(len(X_data) * 0.7)
X_train, y_train = X_data[:split], y_data[:split]
X_test, y_test = X_data[split:], y_data[split:]

# Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(100, 1)),
    LSTM(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1),
])
model.compile(optimizer="adam", loss="mean_squared_error")

if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    st.success("Model training completed!")

    # Predictions on test data
    predictions = model.predict(X_test)
    inv_preds = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_test)

    st.subheader("Test Data Predictions vs Actual Data")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data.index[split+100:], inv_y_test, label="Actual")
    ax.plot(stock_data.index[split+100:], inv_preds, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    # Whole data predictions
    st.subheader("Whole Data Predictions")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data.index, stock_data[price_col], label=f"({price_col}, {stock})", color="blue")
    ax.plot(stock_data.index[-len(y_test):], inv_y_test, label="original test data", color="orange")
    ax.plot(stock_data.index[-len(y_test):], inv_preds, label="preds", color="green")
    ax.set_xlabel("Years")
    ax.set_ylabel("Whole Data")
    ax.set_title("Whole Data of the Stock")
    ax.legend()
    st.pyplot(fig)

    # Future 7 days
    future_days = 7
    last_100 = scaled_data[-100:]
    current_input = last_100.copy()
    future_preds = []

    for _ in range(future_days):
        pred = model.predict(current_input.reshape(1, 100, 1))
        future_preds.append(pred[0, 0])
        current_input = np.append(current_input[1:], pred, axis=0)

    future_preds = np.array(future_preds).reshape(-1, 1)
    inv_future = scaler.inverse_transform(future_preds)
    future_dates = pd.date_range(start=stock_data.index[-1] + pd.DateOffset(1), periods=future_days)

    st.subheader("Future Stock Price Predictions (Next 7 Days)")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(future_dates, inv_future, marker="o", linestyle="dashed", label="Future Prediction")
    ax.legend()
    st.pyplot(fig)

st.write("Adjust parameters and retrain the model to see different results!")

