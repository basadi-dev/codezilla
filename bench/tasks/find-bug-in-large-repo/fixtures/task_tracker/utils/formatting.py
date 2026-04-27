"""Text formatting utilities."""
import re


def truncate(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate text to max_length, appending suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


def format_task_summary(title: str, priority_name: str, days_left: float | None) -> str:
    """Format a one-line task summary."""
    deadline_str = f"{days_left:.0f}d left" if days_left is not None else "no deadline"
    return f"[{priority_name}] {title} ({deadline_str})"
