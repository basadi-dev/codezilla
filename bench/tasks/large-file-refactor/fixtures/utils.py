"""Utility classes — this file has grown too large and should be split up."""
from datetime import datetime
from typing import Any, Optional
import json
import re


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    """Simple in-memory logger with level filtering."""

    LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}

    def __init__(self, min_level: str = "INFO"):
        self.min_level = min_level
        self.entries: list[dict] = []

    def log(self, level: str, message: str) -> None:
        level_val = self.LEVELS.get(level.upper(), 0)
        min_val = self.LEVELS.get(self.min_level.upper(), 0)
        if level_val >= min_val:
            self.entries.append({
                "level": level.upper(),
                "message": message,
                "timestamp": datetime.now().isoformat(),
            })

    def debug(self, message: str) -> None:
        self.log("DEBUG", message)

    def info(self, message: str) -> None:
        self.log("INFO", message)

    def warning(self, message: str) -> None:
        self.log("WARNING", message)

    def error(self, message: str) -> None:
        self.log("ERROR", message)

    def get_entries(self, level: Optional[str] = None) -> list[dict]:
        if level is None:
            return list(self.entries)
        return [e for e in self.entries if e["level"] == level.upper()]

    def clear(self) -> None:
        self.entries.clear()

    def count(self) -> int:
        return len(self.entries)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """Nested configuration with dot-notation access and JSON serialization."""

    def __init__(self, data: Optional[dict] = None):
        self._data: dict = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by dot-separated key path."""
        parts = key.split(".")
        current = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """Set a value by dot-separated key path, creating intermediate dicts."""
        parts = key.split(".")
        current = self._data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def has(self, key: str) -> bool:
        """Check if a key path exists."""
        return self.get(key, _SENTINEL) is not _SENTINEL

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        parts = key.split(".")
        current = self._data
        for part in parts[:-1]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        if parts[-1] in current:
            del current[parts[-1]]
            return True
        return False

    def to_json(self) -> str:
        return json.dumps(self._data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Config":
        return cls(json.loads(json_str))

    def merge(self, other: "Config") -> "Config":
        """Deep merge another config into this one. Other takes precedence."""
        result = Config(json.loads(json.dumps(self._data)))
        _deep_merge(result._data, other._data)
        return result

    def keys(self) -> list[str]:
        """Return all top-level keys."""
        return list(self._data.keys())


_SENTINEL = object()


def _deep_merge(target: dict, source: dict) -> None:
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

class Validator:
    """Data validation with chainable rules."""

    def __init__(self):
        self._rules: list[tuple[str, callable, str]] = []

    def required(self, field: str, message: str = "") -> "Validator":
        msg = message or f"{field} is required"
        self._rules.append((field, lambda v: v is not None and v != "", msg))
        return self

    def min_length(self, field: str, length: int, message: str = "") -> "Validator":
        msg = message or f"{field} must be at least {length} characters"
        self._rules.append((field, lambda v, l=length: isinstance(v, str) and len(v) >= l, msg))
        return self

    def max_length(self, field: str, length: int, message: str = "") -> "Validator":
        msg = message or f"{field} must be at most {length} characters"
        self._rules.append((field, lambda v, l=length: isinstance(v, str) and len(v) <= l, msg))
        return self

    def matches(self, field: str, pattern: str, message: str = "") -> "Validator":
        msg = message or f"{field} does not match required pattern"
        self._rules.append((field, lambda v, p=pattern: isinstance(v, str) and bool(re.match(p, v)), msg))
        return self

    def validate(self, data: dict) -> tuple[bool, list[str]]:
        """Validate data against all rules. Returns (is_valid, error_messages)."""
        errors = []
        for field, check, msg in self._rules:
            value = data.get(field)
            if not check(value):
                errors.append(msg)
        return (len(errors) == 0, errors)


# ─────────────────────────────────────────────────────────────────────────────
# Formatter
# ─────────────────────────────────────────────────────────────────────────────

class Formatter:
    """Text formatting utilities."""

    @staticmethod
    def truncate(text: str, max_length: int, suffix: str = "...") -> str:
        if len(text) <= max_length:
            return text
        return text[: max_length - len(suffix)] + suffix

    @staticmethod
    def slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_]+", "-", text)
        text = re.sub(r"-+", "-", text)
        return text.strip("-")

    @staticmethod
    def wrap(text: str, width: int) -> list[str]:
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + len(current_line) > width and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word)
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    @staticmethod
    def title_case(text: str) -> str:
        minor_words = {"a", "an", "the", "and", "but", "or", "for", "nor",
                       "on", "at", "to", "by", "in", "of", "up"}
        words = text.split()
        result = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in minor_words:
                result.append(word.capitalize())
            else:
                result.append(word.lower())
        return " ".join(result)

    @staticmethod
    def indent(text: str, spaces: int = 4, first_line: bool = True) -> str:
        prefix = " " * spaces
        lines = text.split("\n")
        result = []
        for i, line in enumerate(lines):
            if i == 0 and not first_line:
                result.append(line)
            else:
                result.append(prefix + line if line else line)
        return "\n".join(result)
