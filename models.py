# models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib

@dataclass
class Transaction:
    transaction_id: str
    date: Optional[str]
    description: str
    amount: float
    category: str

@dataclass
class FinancialAccount:
    account_id: str
    institution_name: str
    account_type: str
    current_balance: Optional[float] = None
    last_four: Optional[str] = None
    transactions: List[Transaction] = field(default_factory=list) 
@dataclass
class StatementHistoryItem:
    statement_id: str
    upload_time: datetime
    total_income: float
    total_spent: float
    transactions: List[Transaction] = field(default_factory=list)  # Store actual transactions
    category_breakdown: Dict[str, float] = field(default_factory=dict)  # Store categories

@dataclass
class Goal:
    goal_id: str
    name: str
    target_amount: float
    current_amount: float = 0.0
    target_date: Optional[str] = None

@dataclass
class Recommendation:
    recommendation_id: str
    title: str
    description: str
    generation_date: datetime

@dataclass
class User:
    user_id: str
    name: str
    email: str
    password: str
    accounts: List[FinancialAccount] = field(default_factory=list)
    history: List[StatementHistoryItem] = field(default_factory=list)
    goals: List[Goal] = field(default_factory=list)
    profile_image_b64: Optional[str] = None

   
    preferences: Dict[str, Any] = field(
        default_factory=lambda: {
            "date_display": "transaction",  
            "language": "en",               
            "currency": "CAD",              
        }
    )
    def check_password(self, password: str) -> bool:
        return self.password == password
    
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'name': self.name,
            'email': self.email,
            'password': self.password,
            'accounts': [acc.__dict__ for acc in self.accounts],
            'history': [
                {
                    'statement_id': h.statement_id,
                    'upload_time': h.upload_time.isoformat(),
                    'total_income': h.total_income,
                    'total_spent': h.total_spent,
                    'transactions': [t.__dict__ for t in h.transactions],
                    'category_breakdown': h.category_breakdown
                }
                for h in self.history
            ],
            'goals': [g.__dict__ for g in self.goals],

            # Save new fields
            'profile_image_b64': self.profile_image_b64,
            'preferences': self.preferences
        }
        }
    def check_password(self, pwd: str) -> bool:
        return self.password == pwd
