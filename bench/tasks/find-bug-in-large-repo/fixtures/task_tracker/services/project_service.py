"""Project service — business logic for project management."""
from typing import Optional

from ..models.project import Project
from ..storage.memory import MemoryStorage


class ProjectService:
    """Manages project CRUD operations."""

    def __init__(self, storage: Optional[MemoryStorage] = None):
        self.storage = storage or MemoryStorage()

    def create_project(
        self,
        project_id: str,
        name: str,
        description: str = "",
        owner_id: Optional[str] = None,
    ) -> Project:
        project = Project(
            project_id=project_id,
            name=name,
            description=description,
            owner_id=owner_id,
        )
        self.storage.save_project(project)
        return project

    def get_project(self, project_id: str) -> Optional[Project]:
        return self.storage.get_project(project_id)

    def archive_project(self, project_id: str) -> bool:
        project = self.storage.get_project(project_id)
        if project is None:
            return False
        project.archive()
        self.storage.save_project(project)
        return True

    def list_projects(self, include_archived: bool = False) -> list[Project]:
        projects = self.storage.list_projects()
        if not include_archived:
            projects = [p for p in projects if not p.archived]
        return projects
