"""Search service — filtering and querying tasks."""
from datetime import datetime
from typing import Optional

from ..models.task import Task
from ..models.priority import Priority
from ..storage.memory import MemoryStorage


class SearchService:
    """Provides advanced search and filtering for tasks."""

    def __init__(self, storage: Optional[MemoryStorage] = None):
        self.storage = storage or MemoryStorage()

    def search_by_title(self, query: str) -> list[Task]:
        query_lower = query.lower()
        return [
            t for t in self.storage.list_tasks()
            if query_lower in t.title.lower()
        ]

    def search_by_tags(self, tags: list[str]) -> list[Task]:
        tag_set = set(tags)
        return [
            t for t in self.storage.list_tasks()
            if tag_set.intersection(set(t.tags))
        ]

    def filter_by_priority(
        self,
        min_priority: Priority = Priority.LOW,
        max_priority: Priority = Priority.CRITICAL,
    ) -> list[Task]:
        return [
            t for t in self.storage.list_tasks()
            if min_priority <= t.priority <= max_priority and not t.completed
        ]

    def due_within_days(self, days: int, now: Optional[datetime] = None) -> list[Task]:
        ref = now or datetime.now()
        results = []
        for task in self.storage.list_tasks():
            if task.completed or task.deadline is None:
                continue
            remaining = task.days_until_deadline(ref)
            if remaining is not None and 0 <= remaining <= days:
                results.append(task)
        return results
