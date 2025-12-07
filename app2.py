import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# ---------------------- LOGIN SYSTEM ----------------------

VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Login Page")
    st.write("Please enter your credentials to access the Stock Prediction App.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()   # NEW streamlit rerun
        else:
            st.error("Incorrect username or password")


# ---------------------- MAIN APP ----------------------

def run_main_app():

    st.title("📈 Stock Price Prediction using LSTM")

    # Sidebar controls
    st.sidebar.header("Controls")
    stock = st.sidebar.text_input("Enter Stock Ticker (e.g., GOOG)", "GOOG")
    epochs = st.sidebar.slider("Number of Epochs", 1, 50, 2)
    batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)

    # Date range
    end = dt.datetime.now().date()
    start = st.sidebar.date_input("Start Date", dt.date(end.year - 20, end.month, end.day))
    end = st.sidebar.date_input("End Date", end)

    st.write(f"Fetching data for **{stock}** from **{start}** to **{end}** ...")
    stock_data = yf.download(stock, start=start, end=end)

    if stock_data.empty:
        st.error("No data found for this ticker. Try a different one.")
        return

    # Price column
    if "Adj Close" in stock_data.columns:
        col = "Adj Close"
    elif "Close" in stock_data.columns:
        col = "Close"
    else:
        st.error("Neither 'Adj Close' nor 'Close' found in dataset.")
        return

    # Moving averages
    stock_data["MA_100"] = stock_data[col].rolling(100).mean()
    stock_data["MA_250"] = stock_data[col].rolling(250).mean()
    stock_data["percent_change"] = stock_data[col].pct_change()

    # Plot 1
    st.subheader("Adjusted Closing Price")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data[col])
    st.pyplot(fig)

    # Plot 2: 100 MA
    st.subheader("100-Day Moving Average")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data[col], alpha=0.5)
    ax.plot(stock_data["MA_100"], linestyle="dashed")
    st.pyplot(fig)

    # Plot 3: 250 MA
    st.subheader("250-Day Moving Average")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data[col], alpha=0.5)
    ax.plot(stock_data["MA_250"], linestyle="dotted")
    st.pyplot(fig)

    # Plot 4: Percent Change
    st.subheader("Daily Percentage Change")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data["percent_change"])
    st.pyplot(fig)

    # ------------------ LSTM PREP ------------------

    data = stock_data[[col]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X_data, y_data = [], []
    for i in range(100, len(scaled_data)):
        X_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    X_data, y_data = np.array(X_data), np.array(y_data)

    split = int(len(X_data) * 0.7)
    X_train, y_train = X_data[:split], y_data[:split]
    X_test, y_test = X_data[split:], y_data[split:]

    # ------------------ MODEL ------------------

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(100, 1)),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    # ------------------ TRAIN MODEL ------------------

    if st.sidebar.button("Train Model"):
        with st.spinner("Training... Please wait."):
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        st.success("Model trained successfully!")

        # Predictions
        predictions = model.predict(X_test)
        inv_pred = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_test)

        st.subheader("Test Data: Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(18, 5))
        ax.plot(stock_data.index[split+100:], inv_y_test, label="Actual")
        ax.plot(stock_data.index[split+100:], inv_pred, label="Predicted")
        ax.legend()
        st.pyplot(fig)

        # ------------------ FUTURE 7 DAYS ------------------

        st.subheader("Future Stock Prediction (Next 7 Days)")
        last_100 = scaled_data[-100:]
        future = []

        current_input = last_100.copy()
        for _ in range(7):
            pred = model.predict(current_input.reshape(1, 100, 1))
            future.append(pred[0, 0])
            current_input = np.append(current_input[1:], pred, axis=0)

        future = np.array(future).reshape(-1, 1)
        inv_future = scaler.inverse_transform(future)
        dates = pd.date_range(stock_data.index[-1] + pd.DateOffset(1), periods=7)

        fig, ax = plt.subplots(figsize=(18, 5))
        ax.plot(dates, inv_future, marker='o', linestyle='--')
        st.pyplot(fig)

# ---------------------- APP RUNNER ----------------------

if not st.session_state.logged_in:
    login_page()
else:
    run_main_app()
