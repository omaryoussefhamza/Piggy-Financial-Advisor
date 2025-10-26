import streamlit as st # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from datetime import datetime, date
import random

# Page configuration
st.set_page_config(
    page_title="Piggy - Saud's Smart Finance Assistant",
    page_icon="🐷",
    layout="wide"
)

# Initialize session state for data persistence
if 'transactions' not in st.session_state:
    st.session_state.transactions = [
        {"date": "2025-10-01", "desc": "Uber Eats", "amount": -23.45, "category": "Food"},
        {"date": "2025-10-02", "desc": "Starbucks", "amount": -5.75, "category": "Food"},
        {"date": "2025-10-03", "desc": "Groceries", "amount": -45.30, "category": "Food"},
        {"date": "2025-10-04", "desc": "Gas Station", "amount": -35.00, "category": "Transport"},
        {"date": "2025-10-05", "desc": "Paycheck", "amount": 1200.00, "category": "Income"},
        {"date": "2025-10-06", "desc": "Netflix", "amount": -15.99, "category": "Entertainment"},
        {"date": "2025-10-07", "desc": "Amazon", "amount": -89.99, "category": "Shopping"},
    ]

if 'goals' not in st.session_state:
    st.session_state.goals = [
        {"name": "New Laptop", "target": 1500.00, "saved": 850.00, "deadline": "2025-12-31"},
        {"name": "Emergency Fund", "target": 3000.00, "saved": 1200.00, "deadline": "2025-06-30"},
    ]

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Dummy data
user = {"name": "Saud", "email": "saud@piggy.com"}
accounts = [
    {"type": "Checking", "balance": 1200.50, "institution": "TD Bank"},
    {"type": "Savings", "balance": 3400.00, "institution": "RBC"},
    {"type": "Credit Card", "balance": -450.25, "institution": "Scotia Bank"}
]

# Navigation
st.sidebar.title("🐷 Piggy Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Dashboard", "💳 Transactions", "🎯 Goals", "📊 Reports", "🚨 Alerts"])

# Header
st.title("🐷 Piggy: Saud's Smarter Piggy Bank")
st.caption("CP317 Group 15 - Demo Version (Oct 2025)")

# Dashboard Page
if page == "🏠 Dashboard":
    st.header("💰 Financial Dashboard")
    
    # User welcome
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader(f"Welcome, {user['name']}!")
        st.write(f"Email: {user['email']}")
    
    # Account summary
    st.subheader("📊 Account Summary")
    
    total_balance = sum(account['balance'] for account in accounts)
    cols = st.columns(len(accounts))
    
    for i, account in enumerate(accounts):
        with cols[i]:
            st.metric(
                label=f"{account['type']} - {account['institution']}",
                value=f"${account['balance']:,.2f}",
                delta=None
            )
    
    st.metric("Total Net Worth", f"${total_balance:,.2f}")
    
    # Recent transactions preview
    st.subheader("Recent Transactions")
    if st.session_state.transactions:
        recent_transactions = pd.DataFrame(st.session_state.transactions[-5:])
        st.dataframe(recent_transactions, use_container_width=True)
    else:
        st.info("No transactions yet. Add some transactions to see them here!")

# Transactions Page
elif page == "💳 Transactions":
    st.header("💳 Transaction Management")
    
    # Add transaction form
    with st.form("add_transaction"):
        st.subheader("Add New Transaction")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trans_date = st.date_input("Date", date.today())
            amount = st.number_input("Amount", min_value=-10000.0, max_value=10000.0, step=0.01)
        with col2:
            description = st.text_input("Description", placeholder="Uber Eats, Paycheck, etc.")
            category = st.selectbox("Category", ["Food", "Transport", "Entertainment", "Shopping", "Income", "Bills", "Other"])
        with col3:
            account = st.selectbox("Account", [acc['type'] for acc in accounts])
        
        submitted = st.form_submit_button("Add Transaction")
        
        if submitted and description:
            new_transaction = {
                "date": trans_date.strftime("%Y-%m-%d"),
                "desc": description,
                "amount": amount,
                "category": category,
                "account": account
            }
            st.session_state.transactions.append(new_transaction)
            st.success("Transaction added successfully!")
            st.rerun()
    
    # Display all transactions
    st.subheader("All Transactions")
    if st.session_state.transactions:
        df_transactions = pd.DataFrame(st.session_state.transactions)
        st.dataframe(df_transactions, use_container_width=True)
        
        # Clear transactions button
        if st.button("Clear All Transactions"):
            st.session_state.transactions = []
            st.rerun()
    else:
        st.info("No transactions available. Add some transactions above!")

# Goals Page
elif page == "🎯 Goals":
    st.header("🎯 Financial Goals")
    
    # Add goal form
    with st.form("add_goal"):
        st.subheader("Create New Goal")
        col1, col2 = st.columns(2)
        
        with col1:
            goal_name = st.text_input("Goal Name", placeholder="New Laptop, Vacation, etc.")
            target_amount = st.number_input("Target Amount ($)", min_value=1.0, step=1.0)
        with col2:
            current_saved = st.number_input("Currently Saved ($)", min_value=0.0, step=1.0)
            deadline = st.date_input("Target Date", min_value=date.today())
        
        submitted = st.form_submit_button("Create Goal")
        
        if submitted and goal_name:
            new_goal = {
                "name": goal_name,
                "target": target_amount,
                "saved": current_saved,
                "deadline": deadline.strftime("%Y-%m-%d")
            }
            st.session_state.goals.append(new_goal)
            st.success("Goal created successfully!")
            st.rerun()
    
    # Display goals with progress
    st.subheader("Your Goals")
    if st.session_state.goals:
        for i, goal in enumerate(st.session_state.goals):
            progress = goal['saved'] / goal['target']
            cols = st.columns([3, 1])
            
            with cols[0]:
                st.write(f"**{goal['name']}**")
                st.progress(progress)
                st.write(f"${goal['saved']:,.2f} / ${goal['target']:,.2f} ({progress:.1%})")
                st.caption(f"Target: {goal['deadline']}")
            
            with cols[1]:
                with st.form(f"update_goal_{i}"):
                    add_amount = st.number_input(f"Add to {goal['name']}", min_value=0.0, step=10.0, key=f"add_{i}")
                    if st.form_submit_button("Add Savings"):
                        st.session_state.goals[i]['saved'] += add_amount
                        st.success(f"Added ${add_amount:,.2f} to {goal['name']}!")
                        st.rerun()
    else:
        st.info("No goals set yet. Create your first financial goal above!")

# Reports Page
elif page == "📊 Reports":
    st.header("📊 Spending Reports")
    
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        
        # Spending by category
        st.subheader("Spending by Category")
        category_spending = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        
        if not category_spending.empty:
            fig, ax = plt.subplots()
            category_spending.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_ylabel('Amount ($)')
            ax.set_title('Spending by Category')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No spending data available for chart.")
        
        # Monthly spending trend
        st.subheader("Transaction History")
        st.dataframe(df, use_container_width=True)
        
        # Basic statistics
        st.subheader("Financial Summary")
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_spending = df[df['amount'] < 0]['amount'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Income", f"${total_income:,.2f}")
        with col2:
            st.metric("Total Spending", f"${abs(total_spending):,.2f}")
        with col3:
            st.metric("Net Cash Flow", f"${total_income + total_spending:,.2f}")
    
    else:
        st.info("No transaction data available for reports. Add some transactions first!")

# Alerts Page
elif page == "🚨 Alerts":
    st.header("🚨 Oink Alerts")
    
    # Generate alerts based on transaction data
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        
        # Check for high food spending
        total_spending = df[df['amount'] < 0]['amount'].sum()
        food_spending = df[(df['category'] == 'Food') & (df['amount'] < 0)]['amount'].sum()
        
        if total_spending != 0 and abs(food_spending) > abs(total_spending) * 0.3:
            st.warning("🍔 **High Food Spending Alert**: You're spending more than 30% of your budget on food. Consider meal planning!")
        
        # Check for high entertainment spending
        entertainment_spending = df[(df['category'] == 'Entertainment') & (df['amount'] < 0)]['amount'].sum()
        if total_spending != 0 and abs(entertainment_spending) > abs(total_spending) * 0.2:
            st.warning("🎬 **Entertainment Budget Alert**: Your entertainment spending is high. Look for free alternatives!")
        
        # Goal progress alerts
        for goal in st.session_state.goals:
            progress = goal['saved'] / goal['target']
            if progress >= 1.0:
                st.success(f"🎉 **Goal Achieved**: You've reached your {goal['name']} goal! Congratulations!")
            elif progress >= 0.8:
                st.info(f"📈 **Goal Progress**: You're 80% towards your {goal['name']} goal! Keep going!")
        
        # Low balance alert
        checking_balance = next((acc['balance'] for acc in accounts if acc['type'] == 'Checking'), 0)
        if checking_balance < 100:
            st.error("💸 **Low Balance Alert**: Your checking account balance is getting low!")
    
    else:
        st.info("No alerts to display. Add some transactions and goals to see personalized recommendations!")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("CP317 Software Engineering - Fall 2025")
st.sidebar.caption("Group 15: Piggy Financial Assistant")
