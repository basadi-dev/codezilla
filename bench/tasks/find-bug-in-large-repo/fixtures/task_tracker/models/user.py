"""User model."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class User:
    """A user who can be assigned tasks."""
    user_id: str
    name: str
    email: str
    role: str = "member"
    active: bool = True
    tags: list[str] = field(default_factory=list)

    def display_name(self) -> str:
        return f"{self.name} <{self.email}>"

    def is_admin(self) -> bool:
        return self.role == "admin"

    def deactivate(self) -> None:
        self.active = False
