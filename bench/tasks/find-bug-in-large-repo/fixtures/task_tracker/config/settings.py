"""Application settings."""
from dataclasses import dataclass


@dataclass
class Settings:
    app_name: str = "Task Tracker"
    version: str = "1.0.0"
    max_tasks_per_user: int = 100
    max_title_length: int = 200
    default_page_size: int = 20
    enable_notifications: bool = True


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
