import streamlit as st # type: ignore
import pandas as pd # type: ignore
import re
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from PyPDF2 import PdfReader # type: ignore

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
    password: str  # plain for demo, do not do this in real apps
    accounts: List[FinancialAccount] = field(default_factory=list)
    history: List[StatementHistoryItem] = field(default_factory=list)

    def check_password(self, pwd: str) -> bool:
        return self.password == pwd

    def get_primary_account(self) -> Optional[FinancialAccount]:
        if self.accounts:
            return self.accounts[0]
        return None


# Fake user datastore
USER_STORE: Dict[str, User] = {
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
        Very simple parser:
        - Each line, look for something that looks like a money amount (10.50 or -12.99)
        - Everything before that is treated as date plus description
        - If the first token looks like YYYY-MM-DD, treat it as a date
        """
        transactions: List[Transaction] = []
        counter = 1

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            match = re.search(r"(-?\d+\.\d{2})", line)
            if not match:
                continue

            amount_str = match.group(1)
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
            tx = Transaction(
                transaction_id=f"tx{counter}",
                date=tx_date,
                description=desc,
                amount=amount,
                category=category,
            )
            transactions.append(tx)
            counter += 1

        return transactions

    @staticmethod
    def auto_categorize(desc: str) -> str:
        d = desc.upper()
        if any(
            word in d
            for word in ["UBER EATS", "EATS", "DOORDASH", "RESTAURANT", "CAFE", "STARBUCKS"]
        ):
            return "Food"
        if any(word in d for word in ["WALMART", "COSTCO", "GROCERY", "SUPERMARKET"]):
            return "Groceries"
        if any(word in d for word in ["NETFLIX", "SPOTIFY", "DISNEY", "SUBSCRIPTION"]):
            return "Entertainment"
        if any(word in d for word in ["UBER", "LYFT", "GAS", "SHELL", "PETRO", "TRANSIT"]):
            return "Transport"
        if any(word in d for word in ["PAYROLL", "SALARY", "PAYCHEQUE", "PAYCHECK"]):
            return "Income"
        if any(word in d for word in ["RENT", "MORTGAGE"]):
            return "Housing"
        if any(word in d for word in ["PAYMENT", "REFUND", "CREDIT"]):
            return "Payment or credit"
        return "Other"


class SpendingAnalyzer:
    @staticmethod
    def analyze(transactions: List[Transaction]) -> Dict[str, Any]:
        """
        Treat this like a credit card statement.

        Default rule:
        - Charges and purchases are spending.
        - Payments, refunds, credits and negative amounts are income or offsets.
        """
        if not transactions:
            return {}

        total_spent = 0.0
        total_income = 0.0
        by_cat: Dict[str, float] = {}

        for t in transactions:
            desc = (t.description or "").upper()
            amt = t.amount

            is_payment_like = any(
                key in desc for key in ["PAYMENT", "REFUND", "CREDIT", "RETURN"]
            )

            # Heuristic:
            # If description looks like a payment or refund, treat positive as income, negative as negative spend.
            # Otherwise, for a typical credit card, positive amounts are charges (spend).
            if is_payment_like:
                if amt > 0:
                    total_income += amt
                else:
                    # Negative payment or refund, treat as reduction of spend.
                    total_spent += abs(amt)
                    if t.category not in by_cat:
                        by_cat[t.category] = 0.0
                    by_cat[t.category] += abs(amt)
            else:
                if amt > 0:
                    # Charge or purchase
                    total_spent += amt
                    if t.category not in by_cat:
                        by_cat[t.category] = 0.0
                    by_cat[t.category] += amt
                else:
                    # Negative amount that is not tagged as payment, treat as income or rebate
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
            text = (
                "Upload a statement so I can review your spending and suggest improvements."
            )
            return Recommendation(
                recommendation_id="rec-empty",
                title="No data",
                description=text,
                generation_date=datetime.now(),
            )

        total_income = analysis["total_income"]
        total_spent = analysis["total_spent"]
        by_cat = analysis["by_category"]

        lines: List[str] = []
        lines.append(
            f"In this statement you spent about ${total_spent:,.2f} and received around ${total_income:,.2f} in income."
        )

        if total_income > 0:
            savings_rate = (total_income - total_spent) / total_income
            lines.append(
                f"Your estimated savings rate is about {savings_rate * 100:.1f} percent."
            )
            if savings_rate < 0:
                lines.append(
                    "You are spending more than you earn in this period. That can be ok in the short term, but long term it leads to debt."
                )
            elif savings_rate < 0.1:
                lines.append(
                    "Your savings rate is quite low. Many advisors suggest aiming for at least 10 to 20 percent of your income."
                )
            else:
                lines.append(
                    "You have a solid savings rate. The important part is to keep it consistent over time."
                )
        else:
            lines.append(
                "I did not detect any income in this statement. This might be a month with only spending or a partial statement."
            )

        if by_cat and total_spent > 0:
            sorted_cats = sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)
            top_cat, top_val = sorted_cats[0]
            share = top_val / total_spent
            lines.append(
                f"Your highest spending category is {top_cat} at about ${top_val:,.2f}, roughly {share * 100:.1f} percent of total spending."
            )

            if share > 0.4:
                lines.append(
                    f"{top_cat} is taking up a large share of your spending. Consider setting a simple monthly cap for this category."
                )
            if "Food" in by_cat and by_cat["Food"] > 0.2 * total_spent:
                lines.append(
                    "Spending on food and eating out looks high. Try planning two or three cheap home meals each week and treat eating out as a planned reward."
                )
            if "Entertainment" in by_cat and by_cat["Entertainment"] > 0.15 * total_spent:
                lines.append(
                    "Entertainment and subscriptions are adding up. Review your subscriptions and cancel anything you rarely use."
                )
            if "Transport" in by_cat and by_cat["Transport"] > 0.15 * total_spent:
                lines.append(
                    "Transport costs stand out. Batch errands, carpool, or use public transit when possible to cut this down."
                )
        else:
            lines.append("I did not find any clear spending categories to comment on.")

        lines.append(
            "As a next step, create a simple monthly budget with three to five categories and compare your next statement to those targets."
        )

        full_text = " ".join(lines)
        # Make sure there are no weird newlines
        full_text = full_text.replace("\n", " ")

        return Recommendation(
            recommendation_id="rec1",
            title="Spending overview and next steps",
            description=full_text,
            generation_date=datetime.now(),
        )

# ===================== SESSION STATE HELPERS =====================


def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "transactions" not in st.session_state:
        st.session_state.transactions = []  # list of Transaction
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "recommendation" not in st.session_state:
        st.session_state.recommendation = None


def get_current_user() -> Optional[User]:
    email = st.session_state.user_email
    if email and email in USER_STORE:
        return USER_STORE[email]
    return None


def login(email: str, password: str) -> bool:
    user = USER_STORE.get(email)
    if not user:
        return False
    if not user.check_password(password):
        return False
    st.session_state.authenticated = True
    st.session_state.user_email = email
    return True


def logout():
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.session_state.transactions = []
    st.session_state.analysis = None
    st.session_state.recommendation = None


def register_user(name: str, email: str, password: str) -> bool:
    """Create a new user, return True if success, False if email already exists."""
    global USER_STORE
    if email in USER_STORE:
        return False

    new_id = f"u{len(USER_STORE) + 1}"
    new_user = User(
        user_id=new_id,
        name=name,
        email=email,
        password=password,
        accounts=[
            FinancialAccount(
                account_id=f"acc{len(USER_STORE) + 1}",
                institution_name="Demo Bank",
                account_type="Credit Card",
            )
        ],
    )
    USER_STORE[email] = new_user
    return True


def require_auth():
    if not st.session_state.authenticated:
        st.warning("Please log in first on the Login page.")
        st.stop()

# ===================== UI PAGES =====================


def render_login_page():
    st.title("Piggy - Your Smarter Piggy Bank")
    st.subheader("Welcome")

    # If already logged in, just tell them and stop
    if st.session_state.authenticated:
        user = get_current_user()
        if user:
            st.success(
                f"You are already logged in as {user.name}. Use the sidebar to navigate."
            )
        else:
            st.success("You are already logged in.")
        return

    col1, col2 = st.columns(2)

    # Login form
    with col1:
        st.subheader("Sign in")
        with st.form("login_form"):
            email = st.text_input("Email", "niya@piggy.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")

        if submitted:
            if login(email, password):
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid email or password.")

    # Sign up form
    with col2:
        st.subheader("Sign up")
        with st.form("signup_form"):
            new_name = st.text_input("Name")
            new_email = st.text_input("New email")
            new_password = st.text_input("New password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            signup_submitted = st.form_submit_button("Create account")

        if signup_submitted:
            if not new_name or not new_email or not new_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif new_email in USER_STORE:
                st.error("An account with this email already exists.")
            else:
                ok = register_user(new_name, new_email, new_password)
                if ok:
                    st.success(
                        "Account created. You can now sign in with your new credentials."
                    )
                else:
                    st.error("Could not create account. Try a different email.")


def render_dashboard_page():
    require_auth()
    user = get_current_user()
    st.title("Dashboard")

    if user:
        st.write(f"Welcome, **{user.name}**.")
    else:
        st.write("Welcome.")

    if not st.session_state.transactions:
        st.info("No statement uploaded yet. Go to the Reports page to upload a statement.")
        return

    tx_list: List[Transaction] = st.session_state.transactions
    df = pd.DataFrame(
        [
            {
                "Date": t.date,
                "Description": t.description,
                "Amount": t.amount,
                "Category": t.category,
            }
            for t in tx_list
        ]
    )

    st.subheader("Recent transactions")
    st.dataframe(df.tail(10))

    analysis = st.session_state.analysis
    if analysis:
        col1, col2 = st.columns(2)
        col1.metric("Total spent", f"${analysis['total_spent']:,.2f}")
        col2.metric("Total income", f"${analysis['total_income']:,.2f}")

        cat_df = pd.DataFrame(
            [{"Category": k, "Spend": v} for k, v in analysis["by_category"].items()]
        )

        if not cat_df.empty:
            cat_df = cat_df.sort_values("Spend", ascending=False)
            st.subheader("Spending by category")
            st.bar_chart(cat_df.set_index("Category"))
        else:
            st.info("No category level spending to display yet.")


def render_reports_page():
    require_auth()
    st.title("Spending report and Piggy AI")

    st.write(
        "Upload a credit card statement in PDF form. I will try to detect transactions, summarize your spending, and give a recommendation."
    )

    uploaded_file = st.file_uploader("Upload credit card statement (PDF)", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Reading and analyzing your statement..."):
            text = PDFStatementParser.extract_text(uploaded_file)
            transactions = PDFStatementParser.parse_transactions(text)

            if not transactions:
                st.error(
                    "I could not detect any transactions. Check the PDF format or try another statement."
                )
                return

            st.session_state.transactions = transactions

            analysis = SpendingAnalyzer.analyze(transactions)
            st.session_state.analysis = analysis

            # Save a snapshot in the current user's history (always, even if zero)
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

        tx_list: List[Transaction] = st.session_state.transactions
        df = pd.DataFrame(
            [
                {
                    "Date": t.date,
                    "Description": t.description,
                    "Amount": t.amount,
                    "Category": t.category,
                }
                for t in tx_list
            ]
        )

        st.subheader("Detected transactions")
        st.dataframe(df.head(30))

    rec: Optional[Recommendation] = st.session_state.recommendation
    if rec:
        st.subheader(rec.title)
        st.write(rec.description)
        st.caption(
            f"Generated at {rec.generation_date.strftime('%Y-%m-%d %H:%M:%S')}"
        )


def render_placeholder_page(title: str, text: str):
    require_auth()
    st.title(title)
    st.info(text)


def render_history_page():
    require_auth()
    user = get_current_user()
    st.title("Statement history")

    if not user or not user.history:
        st.info("No previous statements found for this user.")
        return

    data = []
    for item in user.history:
        data.append(
            {
                "Statement ID": item.statement_id,
                "Uploaded at": item.upload_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Total income": item.total_income,
                "Total spent": item.total_spent,
            }
        )

    df = pd.DataFrame(data).sort_values("Uploaded at", ascending=False)
    st.dataframe(df)

# ===================== MAIN =====================

init_session_state()

# If not authenticated, only show login page, no navigation
if not st.session_state.authenticated:
    st.sidebar.empty()
    render_login_page()
else:
    st.sidebar.title("Piggy navigation")
    user = get_current_user()
    if user:
        st.sidebar.write(f"Logged in as **{user.name}**")

    if st.sidebar.button("Log out", key="sidebar_logout"):
        logout()
        st.rerun()

    page = st.sidebar.radio(
        "Go to:",
        ["Dashboard", "Reports", "History", "Goals (future)", "Settings (future)"],
    )

    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Reports":
        render_reports_page()
    elif page == "History":
        render_history_page()
    elif page == "Goals (future)":
        render_placeholder_page("Goals", "Goal tracking features will be added later.")
    elif page == "Settings (future)":
        render_placeholder_page(
            "Settings", "Account and app settings will be added later."
        )
