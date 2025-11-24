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
    page_title="Piggy - Personal Spending Insights",
    layout="wide"
)

# Piggy brand palette
PRIMARY_PINK = "#fdeef4"      # background pink
PINK_ACTIVE = "#f9b5d0"       # active element
PINK_HOVER = "#fbd7e6"        # hover
NAVY = "#1f2a44"
TEXT_MUTED = "#6b7280"
LIGHT_BG = "#f5f7fb"


def inject_global_styles():
    st.markdown(
        f"""
        <style>
        /* Global font and background */
        html, body, [class*="css"] {{
            font-family: "Calibri", "Segoe UI", sans-serif;
        }}

        .stApp {{
            background-color: {LIGHT_BG};
        }}

        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }}

        /* Sidebar base styling */
        [data-testid="stSidebar"] {{
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }}

        /* Sidebar header branding */
        .piggy-sidebar-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 4px 20px 4px;
            border-bottom: 1px solid #f1f3f5;
            margin-bottom: 12px;
        }}

        .piggy-logo-img {{
            width: 40px;
            height: 40px;
        }}

        .piggy-title {{
            font-size: 22px;
            font-weight: 700;
            color: {NAVY};
        }}

        .piggy-tagline {{
            font-size: 12px;
            color: {TEXT_MUTED};
        }}

        /* Sidebar navigation radio styled as buttons */
        [data-testid="stSidebar"] [role="radiogroup"] > label {{
            display: block;
            width: 100%;
            padding: 10px 14px;
            margin-bottom: 8px;
            border-radius: 10px;
            background-color: {PRIMARY_PINK};
            color: {NAVY};
            border: 1px solid transparent;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
            background-color: {PINK_HOVER};
        }}

        [data-testid="stSidebar"] [role="radiogroup"] > label[aria-checked="true"] {{
            background-color: {PINK_ACTIVE};
            border-color: #e98ab1;
            box-shadow: 0 0 0 1px rgba(233, 138, 177, 0.35);
        }}

        /* Sidebar section titles */
        .piggy-sidebar-section-title {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {TEXT_MUTED};
            margin: 8px 0 4px 2px;
        }}

        .piggy-sidebar-footer {{
            margin-top: 24px;
            padding-top: 12px;
            border-top: 1px solid #f1f3f5;
        }}

        /* Buttons (login, upload, logout etc.) */
        .stButton > button {{
            width: 100%;
            border-radius: 10px;
            background-color: {PINK_ACTIVE};
            color: {NAVY};
            border: 1px solid {PINK_ACTIVE};
            font-weight: 500;
            font-size: 14px;
            padding: 8px 14px;
        }}

        .stButton > button:hover {{
            background-color: {PINK_HOVER};
            border-color: {PINK_HOVER};
        }}

        .stButton > button:active {{
            background-color: #ffffff;
            color: #e98ab1;
            border-color: #e98ab1;
        }}

        /* Metric cards */
        [data-testid="stMetricValue"] {{
            font-size: 1.4rem;
            font-weight: 600;
            color: {NAVY};
        }}

        [data-testid="stMetricLabel"] {{
            font-size: 0.85rem;
            color: {TEXT_MUTED};
        }}

        /* Page titles */
        h1, h2, h3 {{
            color: {NAVY};
        }}

        /* Dataframe tweaks */
        .dataframe th, .dataframe td {{
            font-size: 13px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
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
        accounts_raw = u_data.get("accounts", [])
        accounts = []
        for acc in accounts_raw:
            # Only include fields that FinancialAccount expects
            valid_fields = {
                "account_id": acc.get("account_id"),
                "institution_name": acc.get("institution_name"),
                "account_type": acc.get("account_type"),
                "current_balance": acc.get("current_balance"),
                "last_four": acc.get("last_four"),
            }
            # Remove None values
            valid_fields = {k: v for k, v in valid_fields.items() if v is not None}
            accounts.append(FinancialAccount(**valid_fields))

        history = []
        for h_data in u_data.get("history", []):
            transactions = [
                Transaction(
                    transaction_id=t["transaction_id"],
                    date=t["date"],
                    description=t["description"],
                    amount=t["amount"],
                    category=t["category"],
                )
                for t in h_data.get("transactions", [])
            ]

            history.append(
                StatementHistoryItem(
                    statement_id=h_data["statement_id"],
                    upload_time=datetime.fromisoformat(h_data["upload_time"]),
                    total_income=h_data["total_income"],
                    total_spent=h_data["total_spent"],
                    transactions=transactions,
                    category_breakdown=h_data.get("category_breakdown", {}),
                )
            )

        goals = [Goal(**g) for g in u_data.get("goals", [])]

        loaded[email] = User(
            user_id=u_data["user_id"],
            name=u_data["name"],
            email=u_data["email"],
            password=u_data["password"],
            accounts=accounts,
            history=history,
            goals=goals,
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
           "new charges and credits" section.
        2) If that fails, fall back to the Purchases total from
           the summary page.
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

        return []

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
            desc = line[m.end() :].replace(amount_str, "").strip()

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
                desc = line[m.end() :].strip()

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
            text = "Upload a statement so Piggy can review your spending and suggest improvements."
            return Recommendation(
                recommendation_id="rec-empty",
                title="No data available",
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
            lines.append("No income was detected in this statement.")

        if by_cat and total_spent > 0:
            sorted_cats = sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)
            top_cat, top_val = sorted_cats[0]
            share = top_val / total_spent

            lines.append(
                f"Your highest spending category is {top_cat} at ${top_val:,.2f}, about {share * 100:.1f} percent of your spending."
            )
        else:
            lines.append("No clear categories were found.")

        lines.append("Consider setting a simple monthly budget for your largest categories.")

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
    if "nav_choice" not in st.session_state:
        st.session_state.nav_choice = "Dashboard"


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
            "Gemini API key not found. Add it to .streamlit/secrets.toml under [gemini] API_KEY or set GEMINI_API_KEY in your environment."
        )
        return None

    return api_key


def login(email: str, password: str) -> bool:
    email = normalize_email(email)
    user_store = get_user_store()

    st.session_state.debug_info = f"Looking for: '{email}'\nAvailable users: {list(user_store.keys())}"

    user = user_store.get(email)
    if not user:
        st.session_state.debug_info += f"\nUser not found for email: '{email}'"
        return False
    if not user.check_password(password):
        st.session_state.debug_info += f"\nPassword incorrect for email: '{email}'"
        return False
    st.session_state.authenticated = True
    st.session_state.user_email = email
    st.session_state.debug_info += f"\nLogin successful for: '{email}'"
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
        st.session_state.debug_info = f"Email already exists: '{email}'"
        return False

    new_id = f"u{len(user_store) + 1}"
    new_user = User(
        user_id=new_id,
        name=name,
        email=email,
        password=password,
        accounts=[
            FinancialAccount(
                account_id=f"acc{len(user_store) + 1}",
                institution_name="Demo Bank",
                account_type="Credit Card",
            )
        ],
    )

    user_store[email] = new_user
    st.session_state.user_store = user_store

    save_users_to_file(user_store)

    return True


def require_auth():
    if not st.session_state.authenticated:
        st.warning("Please sign in first.")
        st.stop()


def compute_total_savings_from_history(user: User) -> float:
    """For each statement, add max(income - spending, 0) across history."""
    total = 0.0
    for item in user.history:
        net = item.total_income - item.total_spent
        if net > 0:
            total += net
    return total

# ===================== UI PAGES =====================


def render_login_page():
    st.title("Piggy")
    st.subheader("Personal spending insights")

    # Debug toggle
    if st.checkbox("Show debug information"):
        st.session_state.show_debug = True
    else:
        st.session_state.show_debug = False

    if st.session_state.signup_success:
        st.success(st.session_state.signup_success)
        st.session_state.signup_success = None

    if st.session_state.authenticated:
        user = get_current_user()
        if user:
            st.success(f"You are already logged in as {user.name}.")
        return

    col1, col2 = st.columns(2)

    # Login
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
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid email or password.")
                if st.session_state.show_debug:
                    st.error(f"Debug information:\n{st.session_state.debug_info}")

    # Signup
    with col2:
        st.subheader("Create an account")
        with st.form("signup_form"):
            new_name = st.text_input("Name", placeholder="Your full name")
            new_email = st.text_input("Email", placeholder="your@email.com")
            new_password = st.text_input("Password", type="password", placeholder="Create a password")
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
                    st.session_state.signup_success = (
                        f"Account created successfully for {new_email}. Please sign in with your credentials."
                    )
                    if st.session_state.show_debug:
                        st.success(f"Debug: {st.session_state.debug_info}")
                    st.rerun()
                else:
                    st.error("An account with this email already exists.")
                    if st.session_state.show_debug:
                        st.error(f"Debug: {st.session_state.debug_info}")


def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file using pdfplumber."""
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
    """Generate Gemini-powered insights using full transaction history and goals."""

    if GEMINI_MODEL is None:
        return "Gemini API key is not configured."

    if not user.history:
        return "Upload at least one statement so Piggy can analyze your spending."

    history_sorted = sorted(user.history, key=lambda h: h.upload_time)
    history_data = []
    total_income = 0
    total_spent = 0

    for i, h in enumerate(history_sorted, 1):
        entry = {
            "period": f"Statement {i} - {h.upload_time.strftime('%Y-%m-%d')}",
            "income": h.total_income,
            "spent": h.total_spent,
            "categories": h.category_breakdown,
        }
        history_data.append(entry)
        total_income += h.total_income
        total_spent += h.total_spent

    net_savings = total_income - total_spent
    goals_data = (
        [
            {
                "goal_id": g.goal_id,
                "name": g.name,
                "target_amount": g.target_amount,
                "current_amount": g.current_amount,
                "target_date": g.target_date,
            }
            for g in user.goals
        ]
        if user.goals
        else []
    )

    prompt = f"""
    You are Piggy, a professional but approachable financial coach.

    USER FINANCIAL HISTORY:
    {json.dumps(history_data, indent=2)}

    OVERALL SUMMARY:
    - Total Income: {total_income:.2f}
    - Total Spending: {total_spent:.2f}
    - Net Savings: {net_savings:.2f}

    USER GOALS:
    {json.dumps(goals_data, indent=2)}

    Provide a structured response with:
    1. What the user is doing well
    2. Areas where they can improve spending
    3. Category specific insights
    4. Tips aligned with their goals
    5. A short, encouraging closing paragraph
    Use clear headings and bullet points.
    """

    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"The AI advisor is unavailable at the moment. Error: {str(e)}"


def render_enhanced_ai_feedback():
    require_auth()
    user = get_current_user()

    st.title("AI Financial Advisor")
    st.caption("Personalized financial advice based on your complete spending history.")

    if not user or not user.history:
        st.info("Upload at least one statement in the Reports section to receive AI driven insights.")
        return

    st.subheader("Your Financial Snapshot")

    total_statements = len(user.history)
    total_income = sum(h.total_income for h in user.history)
    total_spent = sum(h.total_spent for h in user.history)
    net_savings = total_income - total_spent

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Statements analyzed", total_statements)
    with col2:
        st.metric("Total income", f"${total_income:,.2f}")
    with col3:
        st.metric("Total spending", f"${total_spent:,.2f}")
    with col4:
        st.metric("Net position", f"${net_savings:,.2f}")

    if st.button("Get personalized financial advice", key="ai_feedback_btn"):
        with st.spinner("Analyzing your financial patterns..."):
            advice = get_enhanced_ai_feedback(user)

        st.subheader("Your personalized financial plan")
        st.markdown("---")
        st.markdown(advice)
        st.markdown("---")
        st.caption(
            "This advice is generated by AI and is for informational purposes only. For major financial decisions, consult a qualified professional."
        )


def render_enhanced_dashboard():
    require_auth()
    user = get_current_user()

    st.title("Financial Dashboard")
    st.caption("Overview of your most recent statement and trends over time.")

    if user:
        st.write(f"Welcome back, **{user.name}**.")
    else:
        st.error("User not found.")
        return

    latest_analysis = None
    latest_transactions = []

    if user.history:
        latest_item = sorted(user.history, key=lambda h: h.upload_time)[-1]
        latest_analysis = {
            "total_income": latest_item.total_income,
            "total_spent": latest_item.total_spent,
            "by_category": latest_item.category_breakdown,
        }
        latest_transactions = latest_item.transactions

    if not latest_analysis and st.session_state.analysis:
        latest_analysis = st.session_state.analysis
        latest_transactions = st.session_state.transactions

    if not latest_analysis:
        st.info("No financial data found yet. Upload a statement in the Reports section to get started.")
        return

    # Key metrics
    st.subheader("Key metrics")

    total_spent = latest_analysis.get("total_spent", 0)
    total_income = latest_analysis.get("total_income", 0)
    by_category = latest_analysis.get("by_category", {})

    net_flow = total_income - total_spent
    savings_rate = (net_flow / total_income * 100) if total_income > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total income", value=f"${total_income:,.2f}")
    with col2:
        st.metric(label="Total spending", value=f"${total_spent:,.2f}")
    with col3:
        delta_color = "normal" if net_flow >= 0 else "inverse"
        st.metric(label="Net cash flow", value=f"${net_flow:,.2f}", delta_color=delta_color)
    with col4:
        status = "Good" if savings_rate >= 20 else "Needs attention"
        st.metric(label="Savings rate", value=f"{savings_rate:.1f} percent", delta=status)

    # Spending visualizations
    if by_category:
        st.subheader("Spending breakdown")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if sum(by_category.values()) > 0:
                fig_pie = px.pie(
                    values=list(by_category.values()),
                    names=list(by_category.keys()),
                    title="Spending by category",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, width="stretch")

        with chart_col2:
            categories = list(by_category.keys())
            amounts = list(by_category.values())

            fig_bar = px.bar(
                x=categories,
                y=amounts,
                title="Spending by category",
                labels={"x": "Category", "y": "Amount ($)"},
                color=amounts,
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, width="stretch")

    # Recent transactions
    st.subheader("Recent transactions")

    if latest_transactions:
        df = pd.DataFrame(
            [
                {
                    "Date": t.date or "N/A",
                    "Description": t.description,
                    "Amount": f"${t.amount:,.2f}",
                    "Category": t.category,
                    "Type": "Income" if t.amount < 0 else "Expense",
                }
                for t in latest_transactions[-10:]
            ]
        )

        st.dataframe(
            df,
            width="stretch",
            column_config={
                "Amount": st.column_config.TextColumn("Amount", help="Transaction amount"),
                "Type": st.column_config.TextColumn("Type", help="Income or Expense"),
            },
        )
    else:
        st.info("No transaction data available for the latest statement.")

    # Trends over time
    if len(user.history) > 1:
        st.subheader("Spending trends")

        history_sorted = sorted(user.history, key=lambda h: h.upload_time)
        dates = [h.upload_time.strftime("%Y-%m-%d") for h in history_sorted]
        spending = [h.total_spent for h in history_sorted]
        income = [h.total_income for h in history_sorted]

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=dates,
                y=spending,
                mode="lines+markers",
                name="Spending",
                line=dict(color="#ff6b6b", width=3),
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=dates,
                y=income,
                mode="lines+markers",
                name="Income",
                line=dict(color="#51cf66", width=3),
            )
        )

        fig_trend.update_layout(
            title="Income and spending over time",
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            hovermode="x unified",
        )

        st.plotly_chart(fig_trend, width="stretch")


def render_enhanced_reports_page():
    require_auth()
    st.title("Detailed reports and analysis")

    st.write("Upload your bank or credit card statements for detailed analysis and insights.")

    uploaded_file = st.file_uploader(
        "Choose a statement file",
        type=["pdf", "csv"],
        help="Supported formats: PDF statements or CSV exports.",
    )

    if uploaded_file is not None:
        with st.spinner("Analyzing your statement..."):
            try:
                if uploaded_file.name.lower().endswith(".pdf"):
                    text = PDFStatementParser.extract_text(uploaded_file)
                    transactions = PDFStatementParser.parse_transactions(text)
                else:
                    transactions = CSVStatementParser.parse_transactions(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return

            if not transactions:
                st.error("No transactions could be extracted from this file.")
                return

            st.session_state.transactions = transactions

            analysis = SpendingAnalyzer.analyze(transactions)
            st.session_state.analysis = analysis

            user = get_current_user()
            if user:
                statement_id = f"stmt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                new_statement = StatementHistoryItem(
                    statement_id=statement_id,
                    upload_time=datetime.now(),
                    total_income=analysis.get("total_income", 0),
                    total_spent=analysis.get("total_spent", 0),
                    transactions=transactions,
                    category_breakdown=analysis.get("by_category", {}),
                )

                user.history.append(new_statement)
                user_store = get_user_store()
                save_users_to_file(user_store)

                st.success(f"Statement analyzed and saved successfully. {len(transactions)} transactions found.")

            rec = RecommendationEngine.generate(analysis)
            st.session_state.recommendation = rec

        # Detailed analysis
        st.subheader("Detailed analysis")

        col1, col2 = st.columns(2)

        with col1:
            min_amount = st.number_input("Minimum amount filter", value=0.0, step=10.0)
        with col2:
            category_filter = st.selectbox("Category filter", ["All"] + list(set(t.category for t in transactions)))

        filtered_tx = transactions
        if min_amount > 0:
            filtered_tx = [t for t in filtered_tx if abs(t.amount) >= min_amount]
        if category_filter != "All":
            filtered_tx = [t for t in filtered_tx if t.category == category_filter]

        if filtered_tx:
            df = pd.DataFrame(
                [
                    {
                        "Date": t.date or "Unknown",
                        "Description": t.description,
                        "Amount": t.amount,
                        "Category": t.category,
                        "Type": "Credit" if t.amount < 0 else "Debit",
                    }
                    for t in filtered_tx
                ]
            )

            st.dataframe(df, width="stretch")

            st.subheader("Summary statistics")
            cols = st.columns(3)

            with cols[0]:
                positive = df[df["Amount"] > 0]["Amount"]
                avg_spend = positive.mean() if len(positive) > 0 else 0
                st.metric("Average transaction", f"${avg_spend:.2f}")

            with cols[1]:
                largest_tx = df.loc[df["Amount"].idxmax()] if len(df) > 0 else None
                if largest_tx is not None:
                    st.metric("Largest transaction", f"${largest_tx['Amount']:.2f}", largest_tx["Category"])

            with cols[2]:
                category_count = df["Category"].nunique()
                st.metric("Categories used", category_count)

        if st.session_state.recommendation:
            st.subheader("Personalized recommendations")

            rec = st.session_state.recommendation
            st.info(rec.title)
            st.write(rec.description)
            st.caption(
                f"Generated on {rec.generation_date.strftime('%B %d, %Y at %H:%M')}."
            )


def render_history_page():
    require_auth()
    user = get_current_user()
    st.title("Statement history")

    if not user or not user.history:
        st.info("No previous statements found.")
        return

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

            with cols[3]:
                if st.button("Delete", key=f"delete_{item.statement_id}"):
                    user.history = [h for h in user.history if h.statement_id != item.statement_id]
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

    if not user.history:
        st.info("Upload at least one statement in the Reports section so Piggy can estimate your savings.")
        return

    total_savings = compute_total_savings_from_history(user)

    st.write(
        f"Based on the statements you have uploaded so far, Piggy estimates your total savings at about **${total_savings:,.2f}**."
    )

    if not user.goals:
        st.subheader("Create your first goal")
        with st.form("create_goal_form"):
            goal_name = st.text_input("Goal name", placeholder="Trip, emergency fund, new laptop")
            target_amount_str = st.text_input("Target amount", placeholder="2000")
            target_date = st.text_input("Target date (optional)", placeholder="YYYY-MM-DD")
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
                    save_users_to_file(get_user_store())
                    st.success("Goal created.")
                    st.rerun()
            except ValueError:
                st.error("Please enter a numeric target amount.")
        return

    goal = user.goals[0]
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
            st.info(f"You have reached about {progress * 100:.1f} percent of this goal.")
        else:
            st.success("You have reached this goal.")

    with cols[1]:
        st.markdown(
            f"""
            <div style="
                background-color: #ffffff;
                border-radius: 12px;
                padding: 16px 18px;
                border: 1px solid #f1f3f5;
                color: {NAVY};
                font-size: 14px;">
                <strong>Goal summary</strong><br/>
                Target amount: ${goal.target_amount:,.2f}<br/>
                Current estimated savings: ${goal.current_amount:,.2f}<br/>
                Completion: {progress * 100:.1f} percent
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("----")
    st.subheader("Adjust goal")

    with st.form("update_goal_form"):
        new_name = st.text_input("Goal name", value=goal.name)
        new_target_str = st.text_input("Target amount", value=str(goal.target_amount))
        new_target_date = st.text_input("Target date (optional)", value=goal.target_date or "")
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
        new_email = st.text_input("Email", value=user.email, disabled=True)
        submitted = st.form_submit_button("Save changes")

    if submitted:
        user.name = new_name or user.name
        st.success("Profile updated for this session.")


def render_placeholder_page(title: str, text: str):
    require_auth()
    st.title(title)
    st.info(text)

# ===================== SIDEBAR NAVIGATION =====================


def render_sidebar(user_name: Optional[str]) -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="piggy-sidebar-header">
                <img src="/mnt/data/12707622.png" class="piggy-logo-img" alt="Piggy logo" />
                <div>
                    <div class="piggy-title">Piggy</div>
                    <div class="piggy-tagline">Personal spending insights</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if user_name:
            st.markdown(f"<div style='font-size:12px;color:{TEXT_MUTED};margin-bottom:12px;'>Logged in as <strong>{user_name}</strong></div>", unsafe_allow_html=True)

        st.markdown('<div class="piggy-sidebar-section-title">Navigation</div>', unsafe_allow_html=True)

        nav_choice = st.radio(
            "Navigation",
            ["Dashboard", "Reports", "History", "Goals", "Settings", "AI Feedback"],
            index=["Dashboard", "Reports", "History", "Goals", "Settings", "AI Feedback"].index(
                st.session_state.nav_choice
            ),
            label_visibility="collapsed",
        )

        st.markdown('<div class="piggy-sidebar-footer"></div>', unsafe_allow_html=True)

        if st.button("Log out"):
            logout()
            st.rerun()

    st.session_state.nav_choice = nav_choice
    return nav_choice

# ===================== MAIN =====================

init_session_state()
inject_global_styles()

if not st.session_state.authenticated:
    render_login_page()
else:
    user = get_current_user()
    user_name = user.name if user else None

    nav = render_sidebar(user_name)

    if nav == "Dashboard":
        render_enhanced_dashboard()
    elif nav == "Reports":
        render_enhanced_reports_page()
    elif nav == "History":
        render_history_page()
    elif nav == "Goals":
        render_goals_page()
    elif nav == "Settings":
        render_settings_page()
    elif nav == "AI Feedback":
        render_enhanced_ai_feedback()

    st.markdown(
        "<div style='margin-top:32px;font-size:11px;color:#9ca3af;'>© 2025 Piggy · Demo application for CP317</div>",
        unsafe_allow_html=True,
    )
