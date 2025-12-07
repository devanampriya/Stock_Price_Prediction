import streamlit as st

# --------------- SIMPLE LOGIN ---------------

VALID_USERNAME = "admin"
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
            st.rerun()      # FIXED — replaces experimental_rerun
        else:
            st.error("Incorrect username or password")

if not st.session_state.logged_in:
    login_page()
    st.stop()
