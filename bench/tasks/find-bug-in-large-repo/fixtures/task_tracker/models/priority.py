"""Priority levels for tasks."""
from enum import IntEnum


class Priority(IntEnum):
    """Task priority, higher value = more important."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
