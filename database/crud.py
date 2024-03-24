from .database import SessionLocal, Transaction
from datetime import datetime

def create_transaction(model_id: int, method: str, url: str, status_code: int):
    db = SessionLocal()
    db_transaction = Transaction(model_id=model_id, method=method, url=url, status_code=status_code)
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    db.close()
    return db_transaction

def get_transactions():
    db = SessionLocal()
    transactions = db.query(Transaction).all()
    db.close()
    return transactions