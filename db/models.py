import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Float, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    features = Column(JSONB, nullable=False)
    if_score = Column(Float, nullable=False)
    ae_score = Column(Float, nullable=False)
    final_score = Column(Float, nullable=False)
    predicted_label = Column(Integer, nullable=False)
    true_label = Column(Integer, nullable=True)
    scored_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
