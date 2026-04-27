"""Shared test fixtures."""
import pytest
from datetime import datetime, timedelta

from task_tracker.storage.memory import MemoryStorage
from task_tracker.services.task_service import TaskService
from task_tracker.services.project_service import ProjectService
from task_tracker.models.priority import Priority


@pytest.fixture
def storage():
    return MemoryStorage()


@pytest.fixture
def task_service(storage):
    return TaskService(storage=storage)


@pytest.fixture
def project_service(storage):
    return ProjectService(storage=storage)


@pytest.fixture
def now():
    """A fixed reference time for deterministic tests."""
    return datetime(2025, 6, 15, 12, 0, 0)
