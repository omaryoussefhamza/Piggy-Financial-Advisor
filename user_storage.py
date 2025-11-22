import json
import os
from dataclasses import asdict
from typing import Dict
from datetime import datetime
from models import User, FinancialAccount, StatementHistoryItem, Goal, Transaction

USER_DATA_FILE = "users.json"


def save_users_to_file(user_store: Dict[str, User]):
    """Save users to a JSON file for persistence."""
    serializable = {}

    for email, user in user_store.items():
        serializable[email] = asdict(user)

    with open(USER_DATA_FILE, "w") as f:
        json.dump(serializable, f, indent=4)


def load_users_from_file():
    if not os.path.exists(USER_DATA_FILE):
        return None

    with open(USER_DATA_FILE, "r") as f:
        data = json.load(f)

    loaded = {}

    for email, u in data.items():
        loaded[email] = User(
            user_id=u["user_id"],
            name=u["name"],
            email=u["email"],
            password=u["password"],

            # ACCOUNTS (deep reconstruction)
            accounts=[
                FinancialAccount(
                    account_id=a["account_id"],
                    institution_name=a["institution_name"],
                    account_type=a["account_type"],
                    current_balance=a.get("current_balance", 0.0),

                    transactions=[
                        Transaction(
                            transaction_id=t["transaction_id"],
                            date=t.get("date"),
                            description=t["description"],
                            amount=t["amount"],
                            category=t["category"],
                        )
                        for t in a.get("transactions", [])
                    ],
                )
                for a in u.get("accounts", [])
            ],

            # HISTORY
            history=[
                StatementHistoryItem(
                    statement_id=h["statement_id"],
                    upload_time=datetime.fromisoformat(h["upload_time"]),
                    total_income=h["total_income"],
                    total_spent=h["total_spent"],
                )
                for h in u.get("history", [])
            ],

            # GOALS
            goals=[
                Goal(
                    goal_id=g["goal_id"],
                    name=g["name"],
                    target_amount=g["target_amount"],
                    current_amount=g["current_amount"],
                    target_date=g.get("target_date"),
                )
                for g in u.get("goals", [])
            ],
        )

    return loaded

