import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import re
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import copy

from PyPDF2 import PdfReader  # type: ignore

# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="Piggy - Your Smarter Piggy Bank",
    layout="wide"
)

# Piggy colour palette (soft pink accent)
PRIMARY_COLOR = "#f97373"


def inject_global_styles():
    """Injects global CSS to make the app look more polished."""
    st.markdown(
        f"""
        <style>
        :root {{
            --piggy-primary: {PRIMARY_COLOR};
        }}

        /* Overall page tweaks */
        .block-container {{
            max-width: 1100px;
            padding-top: 3rem;              /* more space at top so logo isn't cut */
        }}

        /* Header bar */
        .piggy-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem 1.0rem 0.7rem 1.0rem;
            margin-top: 0.2rem;
            margin-bottom: 0.8rem;
            border-radius: 0.75rem;
            background: #fff7f7;            /* light pink background to stand out */
            border: 1px solid #ffe4e4;
        }}
        .piggy-brand {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}
        .piggy-logo {{
            font-size: 1.7rem;              /* slightly bigger pig */
            line-height: 1;
        }}
        .piggy-title {{
            font-weight: 700;
            font-size: 1.25rem;
        }}
        .piggy-tagline {{
            font-size: 0.85rem;
            color: #6b7280;
            margin-left: 0.4rem;
        }}
        .piggy-user {{
            font-size: 0.85rem;
            color: #4b5563;
        }}

        /* KPI cards row */
        .piggy-kpi-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.9rem;
            margin-top: 0.6rem;
            margin-bottom: 0.6rem;
        }}
        .piggy-kpi-card {{
            flex: 1 1 200px;
            background: #f9fafb;
            border-radius: 0.8rem;
            border: 1px solid #e5e7eb;
            padding: 0.75rem 1rem;
        }}
        .piggy-kpi-label {{
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 0.2rem;
        }}
        .piggy-kpi-value {{
            font-size: 1.1rem;
            font-weight: 600;
        }}
        .piggy-kpi-sub {{
            font-size: 0.75rem;
            color: #9ca3af;
            margin-top: 0.1rem;
        }}

        /* Tabs styling */
        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: var(--piggy-primary);
            border-bottom: 2px solid var(--piggy-primary);
        }}

        /* Footer */
        .piggy-footer {{
            margin-top: 1.5rem;
            padding-bottom: 0.6rem;
            font-size: 0.75rem;
            color: #9ca3af;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===================== DOMAIN MODEL =====================

@dataclass
class Transaction:
    transaction_id: str
    date: Optional[str]
    description: str
    amount: float
    category: str = "Other"

    def is_debit(self) -> bool:
        return self.amount < 0


@dataclass
class SpendingCategory:
    category_id: str
    name: str
    monthly_budget: Optional[float] = None


@dataclass
class Recommendation:
    recommendation_id: str
    title: str
    description: str
    generation_date: datetime


@dataclass
class StatementHistoryItem:
    statement_id: str
    upload_time: datetime
    total_income: float
    total_spent: float


@dataclass
class FinancialAccount:
    account_id: str
    institution_name: str
    account_type: str
    current_balance: float = 0.0
    transactions: List[Transaction] = field(default_factory=list)

    def add_transaction(self, tx: Transaction) -> None:
        self.transactions.append(tx)


@dataclass
class User:
    user_id: str
    name: str
    email: str
    password: str  # plain for demo only
    accounts: List[FinancialAccount] = field(default_factory=list)
    history: List[StatementHistoryItem] = field(default_factory=list)

    def check_password(self, pwd: str) -> bool:
        return self.password == pwd

    def get_primary_account(self) -> Optional[FinancialAccount]:
        if self.accounts:
            return self.accounts[0]
        return None


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


def get_user_store() -> Dict[str, User]:
    """Get the user store from session state, initialize if not exists"""
    if "user_store" not in st.session_state:
        # Deep copy the initial users to avoid reference issues
        st.session_state.user_store = copy.deepcopy(INITIAL_USER_STORE)
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
        Try to parse transactions from PDF text.

        1) First try 3-line blocks:
           Date
           Description
           Amount

        2) If that fails, fall back to old single-line format:
           YYYY-MM-DD SOME DESCRIPTION -123.45
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        transactions: List[Transaction] = []
        counter = 1

        # ---------- PASS 1: 3-line blocks (Date, Description, Amount) ----------
        i = 0
        while i + 2 < len(lines):
            date_candidate = lines[i]

            if re.match(r"\d{4}-\d{2}-\d{2}", date_candidate):
                desc = lines[i + 1]
                amount_line = lines[i + 2]
                amt_match = re.search(r"(-?\d[\d,]*\.\d{2})", amount_line)
                if amt_match:
                    amount_str = amt_match.group(1).replace(",", "")
                    try:
                        amount = float(amount_str)
                    except ValueError:
                        amount = None

                    if amount is not None:
                        category = PDFStatementParser.auto_categorize(desc)
                        transactions.append(
                            Transaction(
                                transaction_id=f"tx{counter}",
                                date=date_candidate,
                                description=desc,
                                amount=amount,
                                category=category,
                            )
                        )
                        counter += 1
                        i += 3
                        continue  # go to next block

            i += 1

        # If we found any in block mode, return them
        if transactions:
            return transactions

        # ---------- PASS 2: single-line fallback ----------
        for line in lines:
            # find amount
            match = re.search(r"(-?\d[\d,]*\.\d{2})", line)
            if not match:
                continue

            amount_str = match.group(1).replace(",", "")
            try:
                amount = float(amount_str)
            except ValueError:
                continue

            before = line[: match.start()].strip()
            parts = before.split()
            tx_date = None
            desc = before

            if parts and re.match(r"\d{4}-\d{2}-\d{2}", parts[0]):
                tx_date = parts[0]
                desc = " ".join(parts[1:]).strip()

            if not desc:
                desc = "(no description)"

            category = PDFStatementParser.auto_categorize(desc)
            transactions.append(
                Transaction(
                    transaction_id=f"tx{counter}",
                    date=tx_date,
                    description=desc,
                    amount=amount,
                    category=category,
                )
            )
            counter += 1

        return transactions

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
    # Initialize user store in session state
    if "user_store" not in st.session_state:
        st.session_state.user_store = copy.deepcopy(INITIAL_USER_STORE)


def get_current_user() -> Optional[User]:
    email = st.session_state.user_email
    if email:
        email = normalize_email(email)
        user_store = get_user_store()
        return user_store.get(email)
    return None


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
        st.session_state.debug_info = f"❌ Email already exists: '{email}'\nAvailable users: {list(user_store.keys())}"
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
    # Store in session state user store
    user_store[email] = new_user
    st.session_state.user_store = user_store  # Ensure it's stored back
    
    st.session_state.debug_info = f"✅ User registered successfully: '{email}'\nAvailable users: {list(user_store.keys())}"
    return True


def require_auth():
    if not st.session_state.authenticated:
        st.warning("Please log in first on the Login page.")
        st.stop()


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


def render_dashboard_page():
    require_auth()
    user = get_current_user()
    st.title("Dashboard")
    st.caption("Quick overview of your latest statement and spending.")

    if user:
        st.write(f"Welcome, **{user.name}**.")
    else:
        st.write("Welcome.")

    if not st.session_state.transactions:
        st.info("No statement uploaded yet. Go to the Reports page to upload a statement.")
        return

    tx_list: List[Transaction] = st.session_state.transactions
    df = pd.DataFrame(
        [{"Date": t.date, "Description": t.description, "Amount": t.amount, "Category": t.category} for t in tx_list]
    )

    analysis = st.session_state.analysis

    # KPI cards row
    if analysis:
        total_spent = analysis["total_spent"]
        total_income = analysis["total_income"]
        savings_pct = None
        if total_income > 0:
            savings_pct = (total_income - total_spent) / total_income * 100

        # Find top spending category
        top_cat_label = "—"
        top_cat_detail = ""
        if analysis["by_category"] and total_spent > 0:
            sorted_cats = sorted(
                analysis["by_category"].items(), key=lambda kv: kv[1], reverse=True
            )
            top_name, top_val = sorted_cats[0]
            share = top_val / total_spent * 100
            top_cat_label = top_name
            top_cat_detail = f"${top_val:,.2f} ({share:.1f}% of spending)"

        kpi_html = f"""
        <div class="piggy-kpi-row">
            <div class="piggy-kpi-card">
                <div class="piggy-kpi-label">Total spent (this statement)</div>
                <div class="piggy-kpi-value">${total_spent:,.2f}</div>
                <div class="piggy-kpi-sub">Includes all card purchases and debits.</div>
            </div>
            <div class="piggy-kpi-card">
                <div class="piggy-kpi-label">Total income detected</div>
                <div class="piggy-kpi-value">${total_income:,.2f}</div>
                <div class="piggy-kpi-sub">Based on credits such as payroll and refunds.</div>
            </div>
            <div class="piggy-kpi-card">
                <div class="piggy-kpi-label">Estimated savings rate</div>
                <div class="piggy-kpi-value">
                    {f"{savings_pct:.1f}%" if savings_pct is not None else "N/A"}
                </div>
                <div class="piggy-kpi-sub">Approximate (income − spending) ÷ income.</div>
            </div>
            <div class="piggy-kpi-card">
                <div class="piggy-kpi-label">Top spending category</div>
                <div class="piggy-kpi-value">{top_cat_label}</div>
                <div class="piggy-kpi-sub">{top_cat_detail}</div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

    st.subheader("Recent transactions")
    st.dataframe(df.tail(10))

    # Simple spending by category chart
    if analysis and analysis["by_category"]:
        cat_df = pd.DataFrame(
            [{"Category": k, "Spend": v} for k, v in analysis["by_category"].items()]
        ).sort_values("Spend", ascending=False)

        st.subheader("Spending by category")
        st.bar_chart(cat_df.set_index("Category"))
    else:
        st.info("No category-level spending to display yet.")


def render_reports_page():
    require_auth()
    st.title("Spending report and Piggy AI")

    st.write("Upload a credit card or bank statement in **PDF or CSV** format.")

    uploaded_file = st.file_uploader("Upload statement (PDF or CSV)", type=["pdf", "csv"])

    if uploaded_file is not None:
        with st.spinner("Reading and analyzing your statement..."):
            try:
                if uploaded_file.name.lower().endswith(".pdf"):
                    text = PDFStatementParser.extract_text(uploaded_file)
                    transactions = PDFStatementParser.parse_transactions(text)
                else:
                    transactions = CSVStatementParser.parse_transactions(uploaded_file)
            except Exception as e:
                st.error(f"Could not read this file: {e}")
                return

            if not transactions:
                st.error("No transactions detected.")
                return

            st.session_state.transactions = transactions
            analysis = SpendingAnalyzer.analyze(transactions)
            st.session_state.analysis = analysis

            # Save history
            user = get_current_user()
            if user is not None:
                item = StatementHistoryItem(
                    statement_id=f"stmt{len(user.history) + 1}",
                    upload_time=datetime.now(),
                    total_income=analysis.get("total_income", 0.0),
                    total_spent=analysis.get("total_spent", 0.0),
                )
                user.history.append(item)

            rec = RecommendationEngine.generate(analysis)
            st.session_state.recommendation = rec

        df = pd.DataFrame(
            [{
                "Date": t.date,
                "Description": t.description,
                "Amount": t.amount,
                "Category": t.category,
            } for t in st.session_state.transactions]
        )

        st.subheader("Detected transactions")
        st.dataframe(df.head(30))

        # Category breakdown table (more forms of data)
        analysis = st.session_state.analysis
        if analysis and analysis["by_category"]:
            st.subheader("Category breakdown")
            total_spent = analysis["total_spent"]
            rows = []
            for cat, val in analysis["by_category"].items():
                share = (val / total_spent * 100) if total_spent > 0 else 0.0
                rows.append(
                    {"Category": cat, "Amount": val, "Share (%)": round(share, 1)}
                )
            cat_df = pd.DataFrame(rows).sort_values("Amount", ascending=False)
            st.dataframe(cat_df)

    rec = st.session_state.recommendation
    if rec:
        st.subheader(rec.title)
        st.write(rec.description)
        st.caption(f"Generated at {rec.generation_date.strftime('%Y-%m-%d %H:%M:%S')}")


def render_history_page():
    require_auth()
    user = get_current_user()
    st.title("Statement history")

    if not user or not user.history:
        st.info("No previous statements found.")
        return

    data = [{
        "Statement ID": item.statement_id,
        "Uploaded at": item.upload_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Total income": item.total_income,
        "Total spent": item.total_spent,
    } for item in user.history]

    df = pd.DataFrame(data).sort_values("Uploaded at", ascending=False)
    st.dataframe(df)


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
                <span class="piggy-logo">🐷</span>
                <span class="piggy-title">Piggy</span>
                <span class="piggy-tagline">our Smarter Piggy Bank</span>
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
inject_global_styles()

if not st.session_state.authenticated:
    # Just show login / signup when logged out
    render_login_page()
else:
    user = get_current_user()
    user_name = user.name if user else None

    # Simple header with brand + user
    render_piggy_header(user_name)

    # Logout button under header
    if st.button("Log out"):
        logout()
        st.rerun()

    st.markdown("---")

    # Tabs navigation (acts like overhead nav)
    tab_dashboard, tab_reports, tab_history, tab_goals, tab_settings = st.tabs(
        ["Dashboard", "Reports", "History", "Goals (future)", "Settings (future)"]
    )

    with tab_dashboard:
        render_dashboard_page()

    with tab_reports:
        render_reports_page()

    with tab_history:
        render_history_page()

    with tab_goals:
        render_placeholder_page("Goals", "Goal tracking features will be added later.")

    with tab_settings:
        render_placeholder_page("Settings", "Account and app settings will be added later.")

    # Footer
    st.markdown(
        "<div class='piggy-footer'>© 2025 Piggy · Demo app for CP317</div>",
        unsafe_allow_html=True,
    )
