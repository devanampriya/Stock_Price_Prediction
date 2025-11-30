# app.py
import streamlit as st
import streamlit_authenticator as stauth
from passlib.context import CryptContext
import sqlite3
from sqlite3 import Connection
from typing import List, Tuple

# ---------------------------
# Configuration / constants
# ---------------------------
DB_PATH = "users.db"   # file created next to your app
COOKIE_NAME = "stock_app_cookie"
# IMPORTANT: replace this with a secure 32+ char random key in production.
# Put it into Streamlit Cloud's "Secrets" or an env var. We'll fall back to this for local dev.
DEFAULT_COOKIE_KEY = "change_this_to_a_random_secure_key_32_chars_plus"

# ---------------------------
# Utilities: DB and password hashing
# ---------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db() -> Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, name TEXT, email TEXT UNIQUE, password TEXT)"
    )
    conn.commit()
    return conn

def create_user_db_entry(conn: Connection, username: str, name: str, email: str, hashed_password: str) -> bool:
    try:
        conn.execute(
            "INSERT INTO users (username, name, email, password) VALUES (?, ?, ?, ?)",
            (username, name, email, hashed_password),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def fetch_users(conn: Connection) -> List[Tuple[str,str,str,str]]:
    """Return list of (username, name, email, password_hash)"""
    cur = conn.cursor()
    cur.execute("SELECT username, name, email, password FROM users")
    return cur.fetchall()

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# ---------------------------
# Build streamlit-authenticator config from DB
# ---------------------------
def build_config_from_db(conn: Connection):
    rows = fetch_users(conn)
    credentials = {"usernames": {}}
    preauthorized_emails = []
    for username, name, email, pw_hash in rows:
        credentials["usernames"][username] = {
            "name": name,
            "email": email,
            "password": pw_hash  # streamlit-authenticator expects hashed passwords
        }
        preauthorized_emails.append(email)
    config = {
        "credentials": credentials,
        "cookie": {"name": COOKIE_NAME, "key": st.secrets.get("cookie_key", DEFAULT_COOKIE_KEY), "expiry_days": 30},
        "preauthorized": {"emails": preauthorized_emails}
    }
    return config

# ---------------------------
# App UI
# ---------------------------
def main():
    st.set_page_config(page_title="Protected Stock App", layout="wide")
    st.title("Protected Stock Predictor — Login required")

    conn = get_db()

    # Build current auth config from DB
    config = build_config_from_db(conn)

    # Create authenticator
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )

    # Tabs for Login / Signup
    tab_sel = st.sidebar.radio("Action", ["Login", "Sign up (new users)"], index=0)

    if tab_sel == "Login":
        name, authentication_status, username = authenticator.login("Login", "main")
        if authentication_status:
            authenticator.logout("Logout", "sidebar")
            st.success(f"Welcome *{name}* — you are authenticated.")
            # ---------------------------
            # >>> PUT YOUR APP'S MAIN CODE HERE <<<
            # Replace the next block with your existing app's UI and logic.
            #
            # Example: show the protected link or the app content you already have.
            # ---------------------------
            st.markdown("---")
            st.header("Your private app content goes here")
            st.write("This content is shown only to authenticated users.")
            st.write("If your app code is in another file, import and call it here.")
            #
            # Example: if your app used streamlit to show stuff, paste that code here.
            #
            # ---------------------------
        elif authentication_status is False:
            st.error("Username/password is incorrect")
        else:
            st.warning("Please enter your username and password")

    else:  # Sign up
        st.header("Create a new account")
        with st.form("signup_form", clear_on_submit=False):
            new_name = st.text_input("Full name")
            new_username = st.text_input("Choose a username")
            new_email = st.text_input("Email address")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account")

        if submitted:
            if not new_name or not new_username or not new_email or not new_password:
                st.error("All fields are required")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                hashed = hash_password(new_password)
                success = create_user_db_entry(conn, new_username.strip(), new_name.strip(), new_email.strip(), hashed)
                if success:
                    st.success("Account created! You can now log in from the Login tab.")
                    st.experimental_rerun()  # reload so login sees the new user
                else:
                    st.error("Username or email already exists. Pick a different username/email.")

if __name__ == "__main__":
    main()
