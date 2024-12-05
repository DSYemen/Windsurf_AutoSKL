from sqlalchemy.orm import Session
from src.database import get_db
from src.models import User

def get_user_by_username(username: str) -> User:
    db = next(get_db())
    return db.query(User).filter(User.username == username).first()
