"""In-memory storage backend."""
from typing import Optional

from ..models.task import Task
from ..models.user import User
from ..models.project import Project


class MemoryStorage:
    """Simple in-memory store for tasks, users, and projects."""

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._users: dict[str, User] = {}
        self._projects: dict[str, Project] = {}

    # ── Tasks ──

    def save_task(self, task: Task) -> None:
        self._tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def delete_task(self, task_id: str) -> bool:
        return self._tasks.pop(task_id, None) is not None

    def list_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    # ── Users ──

    def save_user(self, user: User) -> None:
        self._users[user.user_id] = user

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def list_users(self) -> list[User]:
        return list(self._users.values())

    # ── Projects ──

    def save_project(self, project: Project) -> None:
        self._projects[project.project_id] = project

    def get_project(self, project_id: str) -> Optional[Project]:
        return self._projects.get(project_id)

    def delete_project(self, project_id: str) -> bool:
        return self._projects.pop(project_id, None) is not None

    def list_projects(self) -> list[Project]:
        return list(self._projects.values())
