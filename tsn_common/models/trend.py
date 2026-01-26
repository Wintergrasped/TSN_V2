"""Trend models - captures topic and callsign trends over rolling windows."""

from datetime import datetime

from sqlalchemy import DateTime, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from tsn_common.models.base import Base


class TrendSnapshot(Base):
    """Stores aggregated trend insights created by the analyzer."""

    __tablename__ = "trend_snapshots"

    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Aggregated data
    trending_topics: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    trending_callsigns: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    notable_nets: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)

    def __repr__(self) -> str:
        return f"<TrendSnapshot(window_start={self.window_start}, window_end={self.window_end})>"
