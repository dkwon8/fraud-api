import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Float, Integer, DateTime, String, JSON
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    features = Column(JSON, nullable=False)
    if_score = Column(Float, nullable=False)
    ae_score = Column(Float, nullable=False)
    final_score = Column(Float, nullable=False)
    predicted_label = Column(Integer, nullable=False)
    true_label = Column(Integer, nullable=True)
    scored_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
