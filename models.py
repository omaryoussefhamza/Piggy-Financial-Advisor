from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class Transaction:
    transaction_id: str
    date: Optional[str]
    description: str
    amount: float
    category: str = "Other"

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
class Goal:
    goal_id: str
    name: str
    target_amount: float
    current_amount: float = 0.0
    target_date: Optional[str] = None

@dataclass
class FinancialAccount:
    account_id: str
    institution_name: str
    account_type: str
    current_balance: float = 0.0
    transactions: List[Transaction] = field(default_factory=list)

@dataclass
class User:
    user_id: str
    name: str
    email: str
    password: str
    accounts: List[FinancialAccount] = field(default_factory=list)
    history: List[StatementHistoryItem] = field(default_factory=list)
    goals: List[Goal] = field(default_factory=list)

    def check_password(self, pwd: str) -> bool:
        return self.password == pwd
