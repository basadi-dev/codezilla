"""Project model."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Project:
    """A project that groups related tasks."""
    project_id: str
    name: str
    description: str = ""
    owner_id: Optional[str] = None
    archived: bool = False
    task_ids: list[str] = field(default_factory=list)

    def add_task(self, task_id: str) -> None:
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)

    def remove_task(self, task_id: str) -> bool:
        if task_id in self.task_ids:
            self.task_ids.remove(task_id)
            return True
        return False

    def archive(self) -> None:
        self.archived = True

    def task_count(self) -> int:
        return len(self.task_ids)
