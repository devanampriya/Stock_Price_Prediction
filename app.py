import streamlit as st

# ------------------- LOGIN SYSTEM -------------------
# Simple login using session_state

# Hardcoded username & password (you can change)
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

# Create session variable if not created
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login_page():
    st.title("🔐 Login Page")
    st.subheader("Please enter your credentials")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Incorrect Username or Password")


# If not logged in → show login page
if not st.session_state.logged_in:
    login_page()
    st.stop()

# ------------------- AFTER LOGIN → RUN app2.py CODE -------------------

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

# Sidebar for user inputs
st.sidebar.header("Controls")
stock = st.sidebar.text_input("Enter Stock Ticker (e.g., GOOG)", "GOOG")
epochs = st.sidebar.slider("Epochs", 1, 50, 2)
batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)

# Date selection
end = dt.datetime.now().date()
start = st.sidebar.date_input("Start Date", dt.date(end.year - 20, end.month, end.day))
end = st.sidebar.date_input("End Date", end)

st.write(f"Fetching data for **{stock}** from **{start}** to **{end}** ...")
stock_data = yf.download(stock, start=start, end=end)

if stock_data.empty:
    st.error("No data found. Try different dates or ticker.")
    st.stop()

# Choose a valid price column
if "Adj Close" in stock_data.columns:
    col = "Adj Close"
elif "Close" in stock_data.columns:
    col = "Close"
else:
    st.error("Adjusted Close/Close price column not found.")
    st.stop()


# Moving Averages
stock_data["MA_100"] = stock_data[col].rolling(100).mean()
stock_data["MA_250"] = stock_data[col].rolling(250).mean()
stock_data["percent_change"] = stock_data[col].pct_change()

# Plot 1: Adjusted Close
st.subheader("Adjusted Closing Price")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data[col], label="Adjusted Close")
ax.legend()
st.pyplot(fig)

# Plot 2: 100-day MA
st.subheader("100-Day Moving Average")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data[col], alpha=0.5)
ax.plot(stock_data["MA_100"], linestyle="dashed")
st.pyplot(fig)

# Plot 3: 250-day MA
st.subheader("250-Day Moving Average")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data[col], alpha=0.5)
ax.plot(stock_data["MA_250"], linestyle="dotted")
st.pyplot(fig)

# Plot 4: Percentage Change
st.subheader("Daily Percentage Change")
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(stock_data["percent_change"])
st.pyplot(fig)

# Build dataset for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[[col]])

X_data, y_data = [], []
for i in range(100, len(scaled_data)):
    X_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

X_data = np.array(X_data)
y_data = np.array(y_data)

split = int(len(X_data) * 0.7)
X_train, y_train = X_data[:split], y_data[:split]
X_test, y_test = X_data[split:], y_data[split:]

# LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(100, 1)),
    LSTM(64),
    Dense(32, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

if st.sidebar.button("Train Model"):
    with st.spinner("Training... Please wait."):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    st.success("Model trained successfully!")

    predictions = model.predict(X_test)
    inv_preds = scaler.inverse_transform(predictions)
    inv_y = scaler.inverse_transform(y_test)

    st.subheader("Test Predictions vs Actual")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data.index[split+100:], inv_y, label="Actual")
    ax.plot(stock_data.index[split+100:], inv_preds, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    # Future predictions
    st.subheader("Next 7 Days Forecast")
    future_input = scaled_data[-100:]
    future_preds = []

    for _ in range(7):
        pred = model.predict(future_input.reshape(1, 100, 1))
        future_preds.append(pred[0, 0])
        future_input = np.append(future_input[1:], pred, axis=0)

    future_preds = np.array(future_preds).reshape(-1, 1)
    inv_fut = scaler.inverse_transform(future_preds)
    dates = pd.date_range(stock_data.index[-1] + pd.Timedelta(days=1), periods=7)

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(dates, inv_fut, marker='o')
    ft = inv_fut.flatten()
    st.pyplot(fig)
