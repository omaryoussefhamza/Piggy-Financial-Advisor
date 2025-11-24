import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import re
from io import BytesIO
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import copy
import time

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PyPDF2 import PdfReader  # type: ignore

# ai imports
import pdfplumber
import re
import google.generativeai as genai

# user data imports
from user_storage import load_users_from_file, save_users_to_file
import json
import os
from models import (
    User, FinancialAccount, StatementHistoryItem, Goal, Transaction, Recommendation
)


# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="Piggy - Your Smarter Piggy Bank",
    layout="wide"
)

# Piggy colour palette (soft pink accent)
PRIMARY_COLOR = "#f97373"
ACCENT_DARK = "#1f2933"
BACKGROUND_LIGHT = "#fff8f8"

def inject_global_styles():
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .stApp {
            background: #f8f9fa;
        }
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 1rem;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f1f3f4;
            border-radius: 10px 10px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #4f46e5;
            color: white;
        }
        
        /* Metric cards styling */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ===================== DOMAIN MODEL =====================


# Initial user store
INITIAL_USER_STORE = {
    "niya@piggy.com": User(
        user_id="u1",
        name="Niya",
        email="niya@piggy.com",
        password="test123",
        accounts=[
            FinancialAccount(
                account_id="acc1",
                institution_name="Demo Bank",
                account_type="Credit Card",
            )
        ],
    ),
    "demo@piggy.com": User(
        user_id="u2",
        name="Demo User",
        email="demo@piggy.com",
        password="demo123",
        accounts=[
            FinancialAccount(
                account_id="acc2",
                institution_name="Demo Bank",
                account_type="Credit Card",
            )
        ],
    ),
}


def normalize_email(email: str) -> str:
    return email.strip().lower()

USER_DATA_FILE = "users.json"

def save_users_to_file(user_store):
    """Properly save all user data including transactions and categories"""
    serializable = {}
    
    for email, user in user_store.items():
        serializable[email] = user.to_dict()
    
    with open(USER_DATA_FILE, "w") as f:
        json.dump(serializable, f, indent=2)


def load_users_from_file():
    if not os.path.exists(USER_DATA_FILE):
        return None

    with open(USER_DATA_FILE, "r") as f:
        data = json.load(f)

    loaded = {}

    for email, u_data in data.items():
        # Safely reconstruct accounts, filtering out unexpected fields
        accounts_raw = u_data.get('accounts', [])
        accounts = []
        for acc in accounts_raw:
            # Only include fields that FinancialAccount expects
            valid_fields = {
                'account_id': acc.get('account_id'),
                'institution_name': acc.get('institution_name'),
                'account_type': acc.get('account_type'),
                'current_balance': acc.get('current_balance'),
                'last_four': acc.get('last_four')
                # Skip 'transactions' and any other unexpected fields
            }
            # Remove None values
            valid_fields = {k: v for k, v in valid_fields.items() if v is not None}
            accounts.append(FinancialAccount(**valid_fields))

        # Rest of your load function remains the same...
        history = []
        for h_data in u_data.get('history', []):
            transactions = [
                Transaction(
                    transaction_id=t['transaction_id'],
                    date=t['date'],
                    description=t['description'],
                    amount=t['amount'],
                    category=t['category']
                ) for t in h_data.get('transactions', [])
            ]
            
            history.append(StatementHistoryItem(
                statement_id=h_data['statement_id'],
                upload_time=datetime.fromisoformat(h_data['upload_time']),
                total_income=h_data['total_income'],
                total_spent=h_data['total_spent'],
                transactions=transactions,
                category_breakdown=h_data.get('category_breakdown', {})
            ))

        goals = [Goal(**g) for g in u_data.get('goals', [])]

        loaded[email] = User(
            user_id=u_data['user_id'],
            name=u_data['name'],
            email=u_data['email'],
            password=u_data['password'],
            accounts=accounts,
            history=history,
            goals=goals
        )

    return loaded

def get_user_store() -> Dict[str, User]:
    if "user_store" not in st.session_state:
        loaded = load_users_from_file()

        if loaded is not None:
            st.session_state.user_store = loaded
        else:
            st.session_state.user_store = copy.deepcopy(INITIAL_USER_STORE)
            save_users_to_file(st.session_state.user_store)

    return st.session_state.user_store


# ===================== SERVICES =====================


class PDFStatementParser:
    @staticmethod
    def extract_text(uploaded_file) -> str:
        pdf_bytes = uploaded_file.read()
        pdf_reader = PdfReader(BytesIO(pdf_bytes))
        chunks = []
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            chunks.append(page_text)
        return "\n".join(chunks)

    @staticmethod
    def parse_transactions(text: str) -> List[Transaction]:
        """
        Parse credit card statements in a bank agnostic way.

        1) First, try to read the structured payment section and the
           "new charges and credits" section. This is the layout your
           CIBC style PDF uses (date date merchant, then category line
           with amount).
        2) If that fails, fall back to using the Purchases total from
           the summary page so we at least get the right spend for the
           period.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        charges = PDFStatementParser._parse_new_charges(lines)
        payments = PDFStatementParser._parse_payments(lines)

        txs: List[Transaction] = []
        counter = 1

        # First add payments (stored as negative amounts so they reduce balance)
        for date_str, desc, amount in payments + charges:
            category = PDFStatementParser.auto_categorize(desc)
            txs.append(
                Transaction(
                    transaction_id=f"tx{counter}",
                    date=date_str,
                    description=desc,
                    amount=amount,
                    category=category,
                )
            )
            counter += 1

        if txs:
            return txs

        # Fallback: just use Purchases total from the summary page
        purchases_total = PDFStatementParser._extract_purchases_total(lines)
        if purchases_total is not None:
            return [
                Transaction(
                    transaction_id="summary1",
                    date=None,
                    description="Statement purchases total",
                    amount=purchases_total,
                    category="Other",
                )
            ]

        # Nothing usable
        return []

    # --------- helpers for structured blocks ---------

    @staticmethod
    def _parse_payments(lines: List[str]):
        """Parse the 'Your payments' block (treat as negative amounts)."""
        in_block = False
        results = []

        datepair_re = re.compile(
            r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+"
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}",
            re.IGNORECASE,
        )
        amount_re = re.compile(r"(-?\d[\d,]*\.\d{2})")

        for line in lines:
            if "Your payments" in line:
                in_block = True
                continue
            if in_block and line.startswith("Total payments"):
                break
            if not in_block:
                continue

            m = datepair_re.match(line)
            if not m:
                continue

            am_m = amount_re.search(line)
            if not am_m:
                continue

            amount_str = am_m.group(1).replace(",", "")
            amount = float(amount_str)

            date_part = m.group(0)
            desc = line[m.end():].replace(amount_str, "").strip()

            # store payments as negative spend
            results.append((date_part, desc, -amount))

        return results

    @staticmethod
    def _parse_new_charges(lines: List[str]):
        """
        Parse the 'Your new charges and credits' block.

        Pattern:

            Oct 03 Oct 06 MERCHANT NAME ...
             Category 13.77
        """
        in_block = False
        results = []

        datepair_re = re.compile(
            r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+"
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}",
            re.IGNORECASE,
        )
        amount_re = re.compile(r"(-?\d[\d,]*\.\d{2})")

        i = 0
        while i < len(lines):
            line = lines[i]

            if "Your new charges and credits" in line:
                in_block = True
                i += 1
                continue

            if in_block and line.startswith("Total for "):
                break

            if not in_block:
                i += 1
                continue

            m = datepair_re.match(line)
            if m:
                date_part = m.group(0)
                desc = line[m.end():].strip()

                # find the next line that has an amount on it
                j = i + 1
                while j < len(lines) and not amount_re.search(lines[j]):
                    j += 1
                if j >= len(lines):
                    break

                amount_line = lines[j]
                am_m = amount_re.search(amount_line)
                amount_str = am_m.group(1).replace(",", "")
                amount = float(amount_str)

                results.append((date_part, desc, amount))
                i = j + 1
            else:
                i += 1

        return results

    @staticmethod
    def _extract_purchases_total(lines: List[str]) -> Optional[float]:
        """Grab 'Purchases 310.15' from the summary if it exists."""
        for line in lines:
            if "Purchases" in line:
                m = re.search(r"(-?\d[\d,]*\.\d{2})", line)
                if m:
                    try:
                        return float(m.group(1).replace(",", ""))
                    except ValueError:
                        pass
        return None

    @staticmethod
    def auto_categorize(desc: str) -> str:
        d = desc.upper()
        if any(word in d for word in ["UBER EATS", "EATS", "DOORDASH", "RESTAURANT", "CAFE", "STARBUCKS"]):
            return "Food"
        if any(word in d for word in ["WALMART", "COSTCO", "GROCERY", "SUPERMARKET", "NO FRILLS", "FRESHCO"]):
            return "Groceries"
        if any(word in d for word in ["NETFLIX", "SPOTIFY", "DISNEY", "SUBSCRIPTION"]):
            return "Entertainment"
        if any(word in d for word in ["UBER", "LYFT", "GAS", "SHELL", "PETRO", "TRANSIT", "TAXI"]):
            return "Transport"
        if any(word in d for word in ["PAYROLL", "SALARY", "PAYCHEQUE", "PAYCHECK", "DEPOSIT"]):
            return "Income"
        if any(word in d for word in ["RENT", "MORTGAGE"]):
            return "Housing"
        if any(word in d for word in ["PAYMENT", "REFUND", "CREDIT", "RETURN"]):
            return "Payment or credit"
        return "Other"


class CSVStatementParser:
    @staticmethod
    def parse_transactions(uploaded_file) -> List[Transaction]:
        """
        Parse CSV with columns like:
        - Date / Transaction Date / Posting Date
        - Description / Details / Memo
        - Amount / Amt / Value
        """
        df = pd.read_csv(uploaded_file)

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        def find_col(possible_names):
            for name in possible_names:
                if name in df.columns:
                    return name
            return None

        date_col = find_col(["date", "transaction date", "posting date", "posted date"])
        desc_col = find_col(["description", "details", "memo", "narrative"])
        amt_col = find_col(["amount", "amt", "value", "transaction amount"])

        if not all([date_col, desc_col, amt_col]):
            raise ValueError(
                "CSV must contain date, description, and amount columns (detected columns: "
                + ", ".join(df.columns)
                + ")"
            )

        transactions: List[Transaction] = []
        counter = 1

        for _, row in df.iterrows():
            date_str = str(row.get(date_col, "")).strip()
            desc = str(row.get(desc_col, "")).strip()
            amt_raw = row.get(amt_col, 0)

            try:
                amount = float(str(amt_raw).replace(",", ""))
            except ValueError:
                continue

            if not desc:
                desc = "(no description)"

            category = PDFStatementParser.auto_categorize(desc)

            transactions.append(
                Transaction(
                    transaction_id=f"csv{counter}",
                    date=date_str if date_str else None,
                    description=desc,
                    amount=amount,
                    category=category,
                )
            )
            counter += 1

        return transactions


class SpendingAnalyzer:
    @staticmethod
    def analyze(transactions: List[Transaction]) -> Dict[str, Any]:
        if not transactions:
            return {}

        total_spent = 0.0
        total_income = 0.0
        by_cat: Dict[str, float] = {}

        for t in transactions:
            desc = (t.description or "").upper()
            amt = t.amount
            is_payment_like = any(key in desc for key in ["PAYMENT", "REFUND", "CREDIT", "RETURN"])

            if is_payment_like:
                if amt > 0:
                    total_income += amt
                else:
                    total_spent += abs(amt)
                    by_cat[t.category] = by_cat.get(t.category, 0.0) + abs(amt)
            else:
                if amt > 0:
                    total_spent += amt
                    by_cat[t.category] = by_cat.get(t.category, 0.0) + amt
                else:
                    total_income += abs(amt)

        return {
            "total_income": total_income,
            "total_spent": total_spent,
            "by_category": by_cat,
        }


class RecommendationEngine:
    @staticmethod
    def generate(analysis: Dict[str, Any]) -> Recommendation:
        if not analysis:
            text = "Upload a statement so I can review your spending and suggest improvements."
            return Recommendation(
                recommendation_id="rec-empty",
                title="No data",
                description=text,
                generation_date=datetime.now(),
            )

        total_income = analysis["total_income"]
        total_spent = analysis["total_spent"]
        by_cat = analysis["by_category"]

        lines = []
        lines.append(
            f"In this statement you spent about ${total_spent:,.2f} and received around ${total_income:,.2f} in income."
        )

        if total_income > 0:
            savings_rate = (total_income - total_spent) / total_income
            lines.append(f"Your estimated savings rate is about {savings_rate * 100:.1f} percent.")

            if savings_rate < 0:
                lines.append("You are spending more than you earn in this period.")
            elif savings_rate < 0.1:
                lines.append("Your savings rate is quite low.")
            else:
                lines.append("You have a solid savings rate.")
        else:
            lines.append("I did not detect income in this statement.")

        if by_cat and total_spent > 0:
            sorted_cats = sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)
            top_cat, top_val = sorted_cats[0]
            share = top_val / total_spent

            lines.append(
                f"Your highest spending category is {top_cat} at ${top_val:,.2f}, about {share * 100:.1f}%."
            )
        else:
            lines.append("No clear categories found.")

        lines.append("Try setting a simple monthly budget.")

        full_text = " ".join(lines).replace("\n", " ")

        return Recommendation(
            recommendation_id="rec1",
            title="Spending overview and next steps",
            description=full_text,
            generation_date=datetime.now(),
        )


# ===================== SESSION HELPERS =====================

def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "transactions" not in st.session_state:
        st.session_state.transactions = []
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "recommendation" not in st.session_state:
        st.session_state.recommendation = None
    if "signup_success" not in st.session_state:
        st.session_state.signup_success = None
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = ""
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False


def get_current_user() -> Optional[User]:
    email = st.session_state.user_email
    if email:
        email = normalize_email(email)
        user_store = get_user_store()
        return user_store.get(email)
    return None

def get_gemini_api_key() -> Optional[str]:
    """Return Gemini API key from Streamlit secrets or environment variable."""
    api_key = None

    # Try Streamlit secrets first
    try:
        api_key = st.secrets["gemini"]["API_KEY"]
    except Exception:
        api_key = None

    # Fallback to environment variable if secrets are not set
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.warning(
            "Gemini API key not found. Please add it to .streamlit/secrets.toml "
            'under [gemini] API_KEY = "..." or set GEMINI_API_KEY in your environment.'
        )
        return None

    return api_key
def login(email: str, password: str) -> bool:
    email = normalize_email(email)
    user_store = get_user_store()
    
    # Debug: Show what's in USER_STORE
    st.session_state.debug_info = f"Looking for: '{email}'\nAvailable users: {list(user_store.keys())}"
    
    user = user_store.get(email)
    if not user:
        st.session_state.debug_info += f"\n❌ User not found for email: '{email}'"
        return False
    if not user.check_password(password):
        st.session_state.debug_info += f"\n❌ Password incorrect for email: '{email}'"
        return False
    st.session_state.authenticated = True
    st.session_state.user_email = email
    st.session_state.debug_info += f"\n✅ Login successful for: '{email}'"
    return True


def logout():
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.session_state.transactions = []
    st.session_state.analysis = None
    st.session_state.recommendation = None
    st.session_state.signup_success = None


def register_user(name: str, email: str, password: str) -> bool:
    email = normalize_email(email)
    user_store = get_user_store()

    if email in user_store:
        st.session_state.debug_info = f"❌ Email already exists: '{email}'"
        return False

    new_id = f"u{len(user_store) + 1}"
    new_user = User(
        user_id=new_id,
        name=name,
        email=email,
        password=password,
        accounts=[FinancialAccount(
            account_id=f"acc{len(user_store) + 1}",
            institution_name="Demo Bank",
            account_type="Credit Card",
        )],
    )

    user_store[email] = new_user
    st.session_state.user_store = user_store

    save_users_to_file(user_store)  # ← persist to disk

    return True


def require_auth():
    if not st.session_state.authenticated:
        st.warning("Please log in first on the Login page.")
        st.stop()

def compute_total_savings_from_history(user: User) -> float:
    """
    Very simple rule: for each statement snapshot, take max(income - spending, 0)
    and sum across history. This is used to fill the piggy bank.
    """
    total = 0.0
    for item in user.history:
        net = item.total_income - item.total_spent
        if net > 0:
            total += net
    return total
    
# ===================== UI PAGES =====================

def render_login_page():
    st.title("Piggy - Your Smarter Piggy Bank")
    st.subheader("Welcome")

    # Debug toggle
    if st.checkbox("Show debug info"):
        st.session_state.show_debug = True
    else:
        st.session_state.show_debug = False

    # Show signup success message if it exists
    if st.session_state.signup_success:
        st.success(st.session_state.signup_success)
        # Clear the message after showing it
        st.session_state.signup_success = None

    # Already logged in
    if st.session_state.authenticated:
        user = get_current_user()
        if user:
            st.success(f"You are already logged in as {user.name}.")
        return

    col1, col2 = st.columns(2)

    # ---------------- LOGIN FORM ----------------
    with col1:
        st.subheader("Sign in")
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign in")

        if submitted:
            if not email or not password:
                st.error("Please enter both email and password.")
            elif login(email, password):
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid email or password. Please check your credentials.")
                # Show detailed debug info
                if st.session_state.show_debug:
                    st.error(f"Debug Info:\n{st.session_state.debug_info}")

    # ---------------- SIGNUP FORM ----------------
    with col2:
        st.subheader("Sign up")
        with st.form("signup_form"):
            new_name = st.text_input("Name", placeholder="Your full name")
            new_email = st.text_input("New email", placeholder="your@email.com")
            new_password = st.text_input("New password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm password", type="password", placeholder="Confirm your password")
            signup_submitted = st.form_submit_button("Create account")

        if signup_submitted:
            if not new_name or not new_email or not new_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok = register_user(new_name, new_email, new_password)
                if ok:
                    # Store success message in session state
                    st.session_state.signup_success = f"✅ Account created successfully for {new_email}! Please sign in with your credentials."
                    if st.session_state.show_debug:
                        st.success(f"Debug: {st.session_state.debug_info}")
                    st.rerun()
                else:
                    st.error("An account with this email already exists. Please use a different email.")
                    if st.session_state.show_debug:
                        st.error(f"Debug: {st.session_state.debug_info}")

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a PDF file using pdfplumber.
    """
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"
    """
    Extract text from a PDF file using pdfplumber.
    """
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception:
        return None


def parse_spending(text: str):
    pattern = r"([A-Za-z ]+)\s+\$?(\d+\.\d{2})"
    results = re.findall(pattern, text)

    spending = []
    for category, amount in results:
        spending.append((category.strip(), float(amount)))
    return spending

api_key = get_gemini_api_key()
if api_key:
    genai.configure(api_key=api_key)
    GEMINI_MODEL = genai.GenerativeModel("models/gemini-2.5-flash")
else:
    GEMINI_MODEL = None

def get_enhanced_ai_feedback(user: User):
    """Generate Gemini-powered insights using: 
       - full transaction history 
       - user goals
    """

    if GEMINI_MODEL is None:
        return "Gemini API key not configured."

    if not user.history:
        return "Upload statements first so I can analyze your spending!"

    # Prepare history
    history_sorted = sorted(user.history, key=lambda h: h.upload_time)
    history_data = []
    total_income = 0
    total_spent = 0

    for i, h in enumerate(history_sorted, 1):
        entry = {
            "period": f"Statement {i} - {h.upload_time.strftime('%Y-%m-%d')}",
            "income": h.total_income,
            "spent": h.total_spent,
            "categories": h.category_breakdown
        }
        history_data.append(entry)
        total_income += h.total_income
        total_spent += h.total_spent

    net_savings = total_income - total_spent
    goals_data = [
    {
        "goal_id": g.goal_id,
        "name": g.name,
        "target_amount": g.target_amount,
        "current_amount": g.current_amount,
        "target_date": g.target_date,
    }
    for g in user.goals
    ] if user.goals else []

    prompt = f"""
    You are Piggy, a friendly financial coach.

    USER FINANCIAL HISTORY:
    {json.dumps(history_data, indent=2)}

    OVERALL SUMMARY:
    - Total Income: {total_income:.2f}
    - Total Spending: {total_spent:.2f}
    - Net Savings: {net_savings:.2f}

    USER GOALS:
    {json.dumps(goals_data, indent=2)}

    Provide:
    1. What the user is doing well
    2. Where they can improve spending
    3. Category-specific insights
    4. Tips aligned with their goals
    5. Encouraging tone
    """

    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI is having issues right now: {str(e)}"


# Update the AI Feedback tab:
def render_enhanced_ai_feedback():
    require_auth()
    user = get_current_user()
    
    st.header("🧠 AI Financial Advisor")
    st.write("Get personalized financial advice based on your complete spending history.")

    if not user or not user.history:
        st.info("📊 Upload your first statement in the Reports tab to get personalized financial advice!")
        return

    # Show user's financial snapshot
    st.subheader("Your Financial Snapshot")
    
    total_statements = len(user.history)
    total_income = sum(h.total_income for h in user.history)
    total_spent = sum(h.total_spent for h in user.history)
    net_savings = total_income - total_spent
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Statements Analyzed", total_statements)
    with col2:
        st.metric("Total Income", f"${total_income:,.2f}")
    with col3:
        st.metric("Total Spending", f"${total_spent:,.2f}")
    with col4:
        st.metric("Net Position", f"${net_savings:,.2f}")

    # FIXED: Truly unique key with timestamp
    import time
    unique_key = f"ai_feedback_btn"
    
    if st.button("🎯 Get Personalized Financial Advice", type="primary", key=unique_key):
        with st.spinner("🤔 Analyzing your financial patterns... This may take a moment."):
            advice = get_enhanced_ai_feedback(user)
            
        st.subheader("💡 Your Personalized Financial Plan")
        
        with st.container():
            st.markdown("---")
            st.markdown(advice)
            st.markdown("---")
            
        st.caption("💡 Remember: This is AI-generated advice. Always consult with a qualified financial advisor for major decisions.")

def render_enhanced_dashboard():
    require_auth()
    user = get_current_user()
    
    st.title("📊 Financial Dashboard")
    st.caption("Complete overview of your financial health")

    if user:
        st.write(f"Welcome back, **{user.name}**! Here's your financial snapshot.")
    else:
        st.error("User not found")
        return

    # Get the latest statement data
    latest_analysis = None
    latest_transactions = []
    
    if user.history:
        latest_item = sorted(user.history, key=lambda h: h.upload_time)[-1]
        latest_analysis = {
            "total_income": latest_item.total_income,
            "total_spent": latest_item.total_spent,
            "by_category": latest_item.category_breakdown
        }
        latest_transactions = latest_item.transactions

    # If no history but we have session data, use that
    if not latest_analysis and st.session_state.analysis:
        latest_analysis = st.session_state.analysis
        latest_transactions = st.session_state.transactions

    if not latest_analysis:
        st.info("💡 No financial data found yet. Visit the Reports tab to upload your first statement!")
        return

    # ========== KPI METRICS ==========
    st.subheader("📈 Key Metrics")
    
    total_spent = latest_analysis.get("total_spent", 0)
    total_income = latest_analysis.get("total_income", 0)
    by_category = latest_analysis.get("by_category", {})
    
    # Calculate derived metrics
    net_flow = total_income - total_spent
    savings_rate = (net_flow / total_income * 100) if total_income > 0 else 0
    
    # Create KPI columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Income",
            value=f"${total_income:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Spending",
            value=f"${total_spent:,.2f}",
            delta=None
        )
    
    with col3:
        delta_color = "normal" if net_flow >= 0 else "inverse"
        st.metric(
            label="Net Cash Flow",
            value=f"${net_flow:,.2f}",
            delta_color=delta_color
        )
    
    with col4:
        status = "Good" if savings_rate >= 20 else "Needs Attention"
        st.metric(
            label="Savings Rate",
            value=f"{savings_rate:.1f}%",
            delta=status
        )

    # ========== SPENDING VISUALIZATIONS ==========
    if by_category:
        st.subheader("💰 Spending Breakdown")
        
        # Create two columns for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Pie chart
            if sum(by_category.values()) > 0:
                fig_pie = px.pie(
                    values=list(by_category.values()),
                    names=list(by_category.keys()),
                    title="Spending by Category",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with chart_col2:
            # Bar chart
            categories = list(by_category.keys())
            amounts = list(by_category.values())
            
            fig_bar = px.bar(
                x=categories,
                y=amounts,
                title="Spending by Category",
                labels={'x': 'Category', 'y': 'Amount ($)'},
                color=amounts,
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    # ========== RECENT TRANSACTIONS ==========
    st.subheader("💳 Recent Transactions")
    
    if latest_transactions:
        # Create a styled dataframe
        df = pd.DataFrame([
            {
                "Date": t.date or "N/A",
                "Description": t.description,
                "Amount": f"${t.amount:,.2f}",
                "Category": t.category,
                "Type": "Income" if t.amount < 0 else "Expense"
            }
            for t in latest_transactions[-10:]  # Last 10 transactions
        ])
        
        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Amount": st.column_config.TextColumn(
                    "Amount",
                    help="Transaction amount"
                ),
                "Type": st.column_config.TextColumn(
                    "Type",
                    help="Income or Expense"
                )
            }
        )
    else:
        st.info("No transaction data available.")

    # ========== SPENDING TRENDS OVER TIME ==========
    if len(user.history) > 1:
        st.subheader("📅 Spending Trends")
        
        # Prepare trend data
        history_sorted = sorted(user.history, key=lambda h: h.upload_time)
        dates = [h.upload_time.strftime('%Y-%m-%d') for h in history_sorted]
        spending = [h.total_spent for h in history_sorted]
        income = [h.total_income for h in history_sorted]
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=dates, y=spending, 
            mode='lines+markers', 
            name='Spending',
            line=dict(color='#ff6b6b', width=3)
        ))
        fig_trend.add_trace(go.Scatter(
            x=dates, y=income, 
            mode='lines+markers', 
            name='Income',
            line=dict(color='#51cf66', width=3)
        ))
        
        fig_trend.update_layout(
            title="Income vs Spending Over Time",
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)

def render_enhanced_reports_page():
    require_auth()
    st.title("📋 Detailed Reports & Analysis")
    
    st.write("Upload your bank or credit card statements for detailed analysis and insights.")
    
    uploaded_file = st.file_uploader("Choose a statement file", type=["pdf", "csv"], 
                                   help="Supported formats: PDF statements or CSV exports")
    
    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing your statement... This may take a moment."):
            try:
                if uploaded_file.name.lower().endswith(".pdf"):
                    text = PDFStatementParser.extract_text(uploaded_file)
                    transactions = PDFStatementParser.parse_transactions(text)
                else:
                    transactions = CSVStatementParser.parse_transactions(uploaded_file)
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                return

            if not transactions:
                st.error("No transactions could be extracted from this file.")
                return

            # Store in session
            st.session_state.transactions = transactions
            
            # Analyze spending
            analysis = SpendingAnalyzer.analyze(transactions)
            st.session_state.analysis = analysis
            
            # Save to user history with FULL data
            user = get_current_user()
            if user:
                # Generate unique statement ID
                statement_id = f"stmt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                new_statement = StatementHistoryItem(
                    statement_id=statement_id,
                    upload_time=datetime.now(),
                    total_income=analysis.get("total_income", 0),
                    total_spent=analysis.get("total_spent", 0),
                    transactions=transactions,  # Store all transactions
                    category_breakdown=analysis.get("by_category", {})  # Store categories
                )
                
                user.history.append(new_statement)
                
                # Persist to disk
                user_store = get_user_store()
                save_users_to_file(user_store)
                
                st.success(f"✅ Statement analyzed and saved successfully! ({len(transactions)} transactions found)")

            # Generate recommendation
            rec = RecommendationEngine.generate(analysis)
            st.session_state.recommendation = rec

        # ========== DETAILED ANALYSIS SECTION ==========
        st.subheader("📊 Detailed Analysis")
        
        # Transaction table with filters
        col1, col2 = st.columns(2)
        
        with col1:
            min_amount = st.number_input("Minimum amount", value=0.0, step=10.0)
        with col2:
            category_filter = st.selectbox("Category", ["All"] + list(set(t.category for t in transactions)))
        
        # Filter transactions
        filtered_tx = transactions
        if min_amount > 0:
            filtered_tx = [t for t in filtered_tx if abs(t.amount) >= min_amount]
        if category_filter != "All":
            filtered_tx = [t for t in filtered_tx if t.category == category_filter]
        
        # Display filtered transactions
        if filtered_tx:
            df = pd.DataFrame([
                {
                    "Date": t.date or "Unknown",
                    "Description": t.description,
                    "Amount": t.amount,
                    "Category": t.category,
                    "Type": "Credit" if t.amount < 0 else "Debit"
                }
                for t in filtered_tx
            ])
            
            st.dataframe(df, use_container_width=True)
            
            # Summary stats
            st.subheader("📈 Summary Statistics")
            cols = st.columns(3)
            
            with cols[0]:
                avg_spend = df[df['Amount'] > 0]['Amount'].mean() if len(df[df['Amount'] > 0]) > 0 else 0
                st.metric("Average Transaction", f"${avg_spend:.2f}")
            
            with cols[1]:
                largest_tx = df.loc[df['Amount'].idxmax()] if len(df) > 0 else None
                if largest_tx is not None:
                    st.metric("Largest Transaction", f"${largest_tx['Amount']:.2f}", largest_tx['Category'])
            
            with cols[2]:
                category_count = df['Category'].nunique()
                st.metric("Categories Used", category_count)
        
        # Enhanced recommendation display
        if st.session_state.recommendation:
            st.subheader("💡 Personalized Recommendations")
            
            rec = st.session_state.recommendation
            with st.container():
                st.success(rec.title)
                st.write(rec.description)
                st.caption(f"Generated on {rec.generation_date.strftime('%B %d, %Y at %H:%M')}")


def render_history_page():
    require_auth()
    user = get_current_user()
    st.title("Statement history")

    if not user or not user.history:
        st.info("No previous statements found.")
        return

    # Show most recent first
    history_sorted = sorted(user.history, key=lambda item: item.upload_time, reverse=True)

    for item in history_sorted:
        with st.container():
            cols = st.columns([3, 2, 2, 1])

            with cols[0]:
                st.markdown(
                    f"**{item.statement_id}**  \n"
                    f"{item.upload_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            with cols[1]:
                st.markdown(f"**Total spent:** ${item.total_spent:,.2f}")
            with cols[2]:
                st.markdown(f"**Total income:** ${item.total_income:,.2f}")

            # Delete button for this statement
            with cols[3]:
                if st.button("Delete", key=f"delete_{item.statement_id}"):
                    # Remove from the user's history
                    user.history = [
                        h for h in user.history if h.statement_id != item.statement_id
                    ]

                    # Persist the change
                    save_users_to_file(get_user_store())

                    st.success(f"Deleted {item.statement_id}.")
                    st.rerun()

        st.markdown("---")

def render_goals_page():
    require_auth()
    user = get_current_user()
    st.title("Savings goals")

    if not user:
        st.info("No user loaded.")
        return

    # If user has uploaded no statements, there is nothing to base progress on
    if not user.history:
        st.info("Upload at least one statement in the Reports tab so I can estimate your savings and track a goal.")
        return

    total_savings = compute_total_savings_from_history(user)

    st.write(
        f"Based on the statements you have uploaded so far, "
        f"I estimate your total savings at about **${total_savings:,.2f}**."
    )

    # If no goal exists yet, let the user create one
    if not user.goals:
        st.subheader("Create your first goal")
        with st.form("create_goal_form"):
            goal_name = st.text_input("Goal name", placeholder="Trip, emergency fund, new laptop")
            target_amount_str = st.text_input("Target amount", placeholder="e.g. 2000")
            target_date = st.text_input("Target date (optional)", placeholder="e.g. 2026-12-31")
            create_submitted = st.form_submit_button("Create goal")

        if create_submitted:
            try:
                target_amount = float(target_amount_str)
                if target_amount <= 0:
                    st.error("Target amount must be positive.")
                else:
                    new_goal = Goal(
                        goal_id=f"g{len(user.goals) + 1}",
                        name=goal_name or "My goal",
                        target_amount=target_amount,
                        current_amount=total_savings,
                        target_date=target_date or None,
                    )
                    user.goals.append(new_goal)

                    # persist
                    save_users_to_file(get_user_store())

                    st.success("Goal created.")
                    st.rerun()
            except ValueError:
                st.error("Please enter a numeric target amount.")
        return  # do not draw the rest until goal exists

    # For now we work with the first goal only
    goal = user.goals[0]
    # Update current amount from latest savings estimate
    goal.current_amount = total_savings
    progress = min(goal.current_amount / goal.target_amount, 1.0) if goal.target_amount > 0 else 0.0

    st.subheader("Current goal")
    cols = st.columns(2)

    with cols[0]:
        st.write(f"**Goal:** {goal.name}")
        st.write(f"**Target:** ${goal.target_amount:,.2f}")
        st.write(f"**Saved so far:** ${goal.current_amount:,.2f}")
        if goal.target_date:
            st.write(f"**Target date:** {goal.target_date}")

        st.progress(progress)

        if progress < 1.0:
            st.info(f"Your piggy bank is about {progress * 100:.1f} percent full.")
        else:
            st.success("Your piggy bank is full. You reached this goal!")

    # Piggy bank visual (simple but clear)
    with cols[1]:
        if progress < 0.25:
            stage = "starting out"
            pig = "🐷"
        elif progress < 0.5:
            stage = "about one quarter full"
            pig = "🐷🪙"
        elif progress < 0.75:
            stage = "about half full"
            pig = "🐷🪙🪙"
        elif progress < 1.0:
            stage = "almost full"
            pig = "🐷🪙🪙🪙"
        else:
            stage = "completely full"
            pig = "🐷💰"

        st.markdown(
            f"""
            <div style="text-align:center; font-size:3rem; margin-top:0.5rem;">
                {pig}
            </div>
            <p style="text-align:center; color:#4b5563;">
                Your piggy bank is <b>{stage}</b> based on your uploaded statements.
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("----")
    st.subheader("Adjust goal")

    with st.form("update_goal_form"):
        new_name = st.text_input("Goal name", value=goal.name)
        new_target_str = st.text_input("Target amount", value=str(goal.target_amount))
        new_target_date = st.text_input(
            "Target date (optional)",
            value=goal.target_date or "",
        )
        update_submitted = st.form_submit_button("Update goal")

    if update_submitted:
        try:
            new_target = float(new_target_str)
            if new_target <= 0:
                st.error("Target amount must be positive.")
            else:
                goal.name = new_name or goal.name
                goal.target_amount = new_target
                goal.target_date = new_target_date or None

                save_users_to_file(get_user_store())

                st.success("Goal updated.")
                st.rerun()
        except ValueError:
            st.error("Please enter a numeric target amount.")

def render_settings_page():
    require_auth()
    user = get_current_user()
    st.title("Settings")

    if not user:
        st.info("No user loaded.")
        return

    st.subheader("Profile")

    with st.form("settings_profile_form"):
        new_name = st.text_input("Display name", value=user.name)
        new_email = st.text_input("Email (cannot be changed here)", value=user.email, disabled=True)
        submitted = st.form_submit_button("Save changes")

    if submitted:
        user.name = new_name or user.name
        st.success("Profile updated for this session.")


def render_placeholder_page(title: str, text: str):
    require_auth()
    st.title(title)
    st.info(text)

# ===================== SIMPLE HEADER =====================

def render_piggy_header(user_name: Optional[str]):
    st.markdown(
        f"""
        <div class="piggy-header">
            <div class="piggy-brand">
                <div class="piggy-logo">🐷</div>
                <div class="piggy-title-block">
                    <span class="piggy-title">Piggy</span>
                    <span class="piggy-tagline">Your smarter online piggy bank</span>
                </div>
            </div>
            <div class="piggy-user">
                {"Logged in as <b>" + user_name + "</b>" if user_name else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===================== MAIN =====================

init_session_state()

if not st.session_state.authenticated:
    # Just show login / signup when logged out
    render_login_page()
else:
    user = get_current_user()
    user_name = user.name if user else None

    # Header
    render_piggy_header(user_name)

    # Logout button under header
    if st.button("Log out"):
        logout()
        st.rerun()

    st.markdown("---")

    # Tabs navigation
    tab_dashboard, tab_reports, tab_history, tab_goals, tab_settings, tab_ai = st.tabs(
        ["Dashboard", "Reports", "History", "Goals", "Settings", "AI Feedback"]
    )

    with tab_dashboard:
        render_enhanced_dashboard()

    with tab_reports:
        render_enhanced_reports_page()

    with tab_history:
        render_history_page()

    with tab_goals:
        render_goals_page()

    with tab_settings:
        render_settings_page()

    with tab_ai:
        render_enhanced_ai_feedback()

    # Footer
    st.markdown(
        "<div class='piggy-footer'>© 2025 Piggy · Demo app for CP317</div>",
        unsafe_allow_html=True,
    )
