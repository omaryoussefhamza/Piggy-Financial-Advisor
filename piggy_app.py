import streamlit as st # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from datetime import datetime, date
import random

st.sidebar.title("Piggy Navigation")
page = st.sidebar.radio("Go to:", ["Login", "Dashboard", "Transactions", "Goals", "Reports"])

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Dummy data
user = {"name": "Saud", "email": "saud@piggy.com"}
accounts = [
    {"type": "Checking", "balance": 1200.50, "institution": "TD Bank"},
    {"type": "Savings", "balance": 3400.00, "institution": "RBC"},
    {"type": "Credit Card", "balance": -450.25, "institution": "Scotia Bank"}
]

st.title("Piggy: A Smart Finance Assistant for Gen Z")

# Alerts Page
elif page == "Alerts":
    st.header("Oink Alerts")
    
    # Generate alerts based on transaction data
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        
        # Check for high food spending
        total_spending = df[df['amount'] < 0]['amount'].sum()
        food_spending = df[(df['category'] == 'Food') & (df['amount'] < 0)]['amount'].sum()
        
        if total_spending != 0 and abs(food_spending) > abs(total_spending) * 0.3:
            st.warning("**High Food Spending Alert**: You're spending more than 30% of your budget on food. Consider meal planning!")
        
        # Check for high entertainment spending
        entertainment_spending = df[(df['category'] == 'Entertainment') & (df['amount'] < 0)]['amount'].sum()
        if total_spending != 0 and abs(entertainment_spending) > abs(total_spending) * 0.2:
            st.warning("**Entertainment Budget Alert**: Your entertainment spending is high. Look for free alternatives!")
        
        # Goal progress alerts
        for goal in st.session_state.goals:
            progress = goal['saved'] / goal['target']
            if progress >= 1.0:
                st.success(f"**Goal Achieved**: You've reached your {goal['name']} goal! Congratulations!")
            elif progress >= 0.8:
                st.info(f"**Goal Progress**: You're 80% towards your {goal['name']} goal! Keep going!")
        
        # Low balance alert
        checking_balance = next((acc['balance'] for acc in accounts if acc['type'] == 'Checking'), 0)
        if checking_balance < 100:
            st.error("**Low Balance Alert**: Your checking account balance is getting low!")
    
    else:
        st.info("No alerts to display. Add some transactions and goals to see personalized recommendations!")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("CP317 Software Engineering - Fall 2025")
st.sidebar.caption("Group 15: Piggy Financial Assistant")
