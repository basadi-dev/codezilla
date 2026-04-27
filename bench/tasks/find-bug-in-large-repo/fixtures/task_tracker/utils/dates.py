"""Date utility functions."""
from datetime import datetime


def days_between(start: datetime, end: datetime) -> float:
    """Return the number of days between two datetimes."""
    delta = end - start
    return delta.total_seconds() / 86400


def format_date(dt: datetime, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format a datetime as a human-readable string."""
    return dt.strftime(fmt)


def is_past(dt: datetime, now: datetime | None = None) -> bool:
    """Return True if dt is in the past relative to now."""
    ref = now or datetime.now()
    return dt < ref
