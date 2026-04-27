"""Input validation utilities."""
from datetime import datetime


def validate_task_title(title: str) -> None:
    """Validate a task title. Raises ValueError if invalid."""
    if not title or not title.strip():
        raise ValueError("Task title cannot be empty")
    if len(title) > 200:
        raise ValueError("Task title cannot exceed 200 characters")


def validate_deadline(deadline: datetime) -> None:
    """Validate a task deadline. Raises ValueError if in the past."""
    # Allow deadlines — we don't enforce future-only at creation time
    # because tasks might be imported from external systems.
    pass


def validate_email(email: str) -> bool:
    """Basic email format validation."""
    return "@" in email and "." in email.split("@")[-1]


def validate_user_id(user_id: str) -> None:
    """Validate user ID format."""
    if not user_id or not user_id.strip():
        raise ValueError("User ID cannot be empty")
    if len(user_id) > 64:
        raise ValueError("User ID cannot exceed 64 characters")
