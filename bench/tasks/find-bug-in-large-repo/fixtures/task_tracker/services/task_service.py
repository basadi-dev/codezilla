"""Task service — business logic for task management."""
from datetime import datetime
from typing import Optional

from ..models.task import Task
from ..models.priority import Priority
from ..storage.memory import MemoryStorage
from ..utils.scoring import compute_priority_score
from ..utils.validation import validate_task_title, validate_deadline


class TaskService:
    """Manages task CRUD and priority-based operations."""

    def __init__(self, storage: Optional[MemoryStorage] = None):
        self.storage = storage or MemoryStorage()

    def create_task(
        self,
        task_id: str,
        title: str,
        priority: Priority = Priority.MEDIUM,
        deadline: Optional[datetime] = None,
        assignee_id: Optional[str] = None,
        project_id: Optional[str] = None,
        description: str = "",
        tags: Optional[list[str]] = None,
    ) -> Task:
        validate_task_title(title)
        if deadline is not None:
            validate_deadline(deadline)

        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            deadline=deadline,
            assignee_id=assignee_id,
            project_id=project_id,
            tags=tags or [],
        )
        self.storage.save_task(task)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.storage.get_task(task_id)

    def complete_task(self, task_id: str) -> bool:
        task = self.storage.get_task(task_id)
        if task is None:
            return False
        task.complete()
        self.storage.save_task(task)
        return True

    def delete_task(self, task_id: str) -> bool:
        return self.storage.delete_task(task_id)

    def list_tasks(
        self,
        assignee_id: Optional[str] = None,
        project_id: Optional[str] = None,
        include_completed: bool = False,
    ) -> list[Task]:
        tasks = self.storage.list_tasks()
        if not include_completed:
            tasks = [t for t in tasks if not t.completed]
        if assignee_id:
            tasks = [t for t in tasks if t.assignee_id == assignee_id]
        if project_id:
            tasks = [t for t in tasks if t.project_id == project_id]
        return tasks

    def sorted_by_priority(
        self,
        now: Optional[datetime] = None,
        assignee_id: Optional[str] = None,
        include_completed: bool = False,
    ) -> list[Task]:
        """Return tasks sorted by computed priority score (highest first)."""
        ref = now or datetime.now()
        tasks = self.list_tasks(
            assignee_id=assignee_id,
            include_completed=include_completed,
        )
        scored = [(compute_priority_score(t, ref), t) for t in tasks]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [task for _, task in scored]

    def overdue_tasks(self, now: Optional[datetime] = None) -> list[Task]:
        ref = now or datetime.now()
        return [t for t in self.list_tasks() if t.is_overdue(ref)]
