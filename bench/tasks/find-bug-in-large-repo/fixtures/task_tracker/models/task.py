"""Task model."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .priority import Priority


@dataclass
class Task:
    """A single task with priority, deadline, and assignment info."""
    task_id: str
    title: str
    description: str = ""
    priority: Priority = Priority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    assignee_id: Optional[str] = None
    project_id: Optional[str] = None
    completed: bool = False
    tags: list[str] = field(default_factory=list)

    def complete(self) -> None:
        self.completed = True

    def assign(self, user_id: str) -> None:
        self.assignee_id = user_id

    def is_overdue(self, now: Optional[datetime] = None) -> bool:
        if self.deadline is None or self.completed:
            return False
        ref = now or datetime.now()
        return ref > self.deadline

    def days_until_deadline(self, now: Optional[datetime] = None) -> Optional[float]:
        if self.deadline is None:
            return None
        ref = now or datetime.now()
        delta = self.deadline - ref
        return delta.total_seconds() / 86400
