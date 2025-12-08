import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# ---------------------------------------------------------
#                     🔐 LOGIN SYSTEM
# ---------------------------------------------------------

USERNAME = "admin"
PASSWORD = "1234"

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_screen():
    st.title("🔐 Login Page")
    st.write("Please log in to access the Stock Prediction App.")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Incorrect username or password")

# If not logged in → show login page ONLY
if not st.session_state.logged_in:
    login_screen()
    st.stop()

# ---------------------------------------------------------
#                   📈 MAIN LSTM APPLICATION
# ---------------------------------------------------------

def run_app():

    st.set_page_config(layout="wide")
    st.title("Stock Price Prediction using LSTM")

    # Sidebar for user inputs
    st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    stock = st.sidebar.text_input("Enter Stock Ticker (e.g., GOOG)", "GOOG").strip()
    epochs = st.sidebar.slider("Number of Epochs", 1, 50, 2)
    batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # Dates
    end = dt.datetime.now()
    start = dt.datetime(end.year - 20, end.month, end.day)

    st.write(f"Fetching data for **{stock}** from {start.date()} to {end.date()} ...")

    @st.cache_data(show_spinner=False)
    def fetch_data(sym, start, end):
        df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False, threads=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        if df.empty or not any("close" in str(c).lower() for c in df.columns):
            tkr = yf.Ticker(sym)
            df2 = tkr.history(start=start, end=end, auto_adjust=True)
            if isinstance(df2.columns, pd.MultiIndex):
                df2.columns = df2.columns.get_level_values(-1)
            if (not df2.empty) and (len(df2.columns) >= len(df.columns)):
                df = df2

        close_col = None
        for c in df.columns:
            if "close" in str(c).lower():
                close_col = c
                break

        if close_col is not None and "Adj Close" not in df.columns:
            df["Adj Close"] = df[close_col]

        return df

    stock_data = fetch_data(stock, start, end)

    if stock_data.empty:
        st.error(f"No data found for `{stock}`. Try tickers like AAPL, TCS.NS, RELIANCE.NS.")
        return

    if "Adj Close" not in stock_data.columns:
        candidates = [c for c in stock_data.columns if "close" in str(c).lower()]
        if candidates:
            stock_data["Adj Close"] = stock_data[candidates[0]]

    if "Adj Close" not in stock_data.columns:
        st.error("Price column missing for this stock.")
        return

    # Moving averages
    stock_data["MA_100"] = stock_data["Adj Close"].rolling(100).mean()
    stock_data["MA_250"] = stock_data["Adj Close"].rolling(250).mean()
    stock_data["percent_change"] = stock_data["Adj Close"].pct_change()

    # ---- Charts ----
    st.subheader("Adjusted Closing Price")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data["Adj Close"])
    st.pyplot(fig)

    st.subheader("100-Day Moving Average")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data["Adj Close"], alpha=0.5)
    ax.plot(stock_data["MA_100"], linestyle="dashed")
    st.pyplot(fig)

    st.subheader("250-Day Moving Average")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data["Adj Close"], alpha=0.5)
    ax.plot(stock_data["MA_250"], linestyle="dotted")
    st.pyplot(fig)

    st.subheader("Daily Percentage Change")
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(stock_data["percent_change"])
    st.pyplot(fig)

    # ---- LSTM Preprocessing ----
    adj_close = stock_data[["Adj Close"]].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(adj_close)

    if len(scaled_data) < 101:
        st.error("Not enough data to train. Need at least 101 days.")
        return

    X_data, y_data = [], []
    for i in range(100, len(scaled_data)):
        X_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    X_data, y_data = np.array(X_data), np.array(y_data)
    split = int(len(X_data) * 0.7)
    X_train, y_train = X_data[:split], y_data[:split]
    X_test, y_test = X_data[split:], y_data[split:]

    # ---- LSTM Model ----
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Training
    if st.sidebar.button("Train Model"):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        st.success("Model Training Completed!")

        predictions = model.predict(X_test)
        inv_preds = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_test)

        st.subheader("Test Predictions vs Actual")
        fig, ax = plt.subplots(figsize=(18, 5))
        ax.plot(stock_data.index[split + 100:], inv_y_test, label="Actual")
        ax.plot(stock_data.index[split + 100:], inv_preds, label="Predicted")
        ax.legend()
        st.pyplot(fig)

        # Whole Data
        st.subheader("Whole Data Predictions")
        fig, ax = plt.subplots(figsize=(18, 5))
        ax.plot(stock_data.index, stock_data["Adj Close"], color="blue", label="Actual")
        ax.plot(stock_data.index[-len(y_test):], inv_y_test, color="orange", label="Original Test Data")
        ax.plot(stock_data.index[-len(y_test):], inv_preds, color="green", label="Predictions")
        ax.legend()
        st.pyplot(fig)

        # Future 7-day prediction
        future_days = 7
        last_100 = scaled_data[-100:]
        future_preds = []
        current_input = last_100.copy()

        for _ in range(future_days):
            pred = model.predict(current_input.reshape(1, 100, 1))
            future_preds.append(pred[0][0])
            current_input = np.append(current_input[1:], pred, axis=0)

        future_preds = np.array(future_preds).reshape(-1, 1)
        inv_future_preds = scaler.inverse_transform(future_preds)
        dates = pd.date_range(stock_data.index[-1] + pd.DateOffset(1), periods=future_days)

        st.subheader("Future Stock Price (Next 7 Days)")
        fig, ax = plt.subplots(figsize=(18, 5))
        ax.plot(dates, inv_future_preds, marker="o")
        st.pyplot(fig)

# Run the app
run_app()
