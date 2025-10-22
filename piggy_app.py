import streamlit as st # type: ignore
import pandas as pd # type: ignore

st.sidebar.title("🐷 Piggy Navigation")
page = st.sidebar.radio("Go to:", ["Login", "Dashboard", "Transactions", "Goals", "Reports"])

user = {"name": "Niya", "email": "niya@piggy.com"}
accounts = [{"type": "Checking", "balance": 1200.50}, {"type": "Savings", "balance": 3400.00}]
transactions = [
    {"date": "2025-10-01", "desc": "Uber Eats", "amount": -23.45, "category": "Food"},
    {"date": "2025-10-05", "desc": "Paycheck", "amount": 800.00, "category": "Income"},
]

st.title("🐷 Piggy: A Smart Finance Assistant for Gen Z")

if page == "Dashboard":
    st.subheader("User Info")
    st.write(user)
    st.subheader("Accounts")
    st.dataframe(pd.DataFrame(accounts))
    st.subheader("Transactions")
    st.dataframe(pd.DataFrame(transactions))

