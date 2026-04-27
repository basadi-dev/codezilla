"""API request handlers."""
from datetime import datetime
from typing import Any

from ..models.priority import Priority
from ..services.task_service import TaskService


def handle_create_task(data: dict[str, Any], service: TaskService) -> dict:
    """Handle a create-task request."""
    title = data.get("title", "")
    priority_str = data.get("priority", "MEDIUM").upper()
    deadline_str = data.get("deadline")

    try:
        priority = Priority[priority_str]
    except KeyError:
        return {"error": f"Invalid priority: {priority_str}"}

    deadline = None
    if deadline_str:
        try:
            deadline = datetime.fromisoformat(deadline_str)
        except ValueError:
            return {"error": f"Invalid deadline format: {deadline_str}"}

    task_id = data.get("id", f"task-{datetime.now().timestamp()}")
    task = service.create_task(
        task_id=task_id,
        title=title,
        priority=priority,
        deadline=deadline,
        description=data.get("description", ""),
    )
    return {"ok": True, "task_id": task.task_id}


def handle_list_tasks(
    data: dict[str, Any],
    service: TaskService,
) -> dict:
    """Handle a list-tasks request."""
    assignee = data.get("assignee_id")
    include_done = data.get("include_completed", False)
    sort_by = data.get("sort_by", "priority")

    if sort_by == "priority":
        tasks = service.sorted_by_priority(assignee_id=assignee, include_completed=include_done)
    else:
        tasks = service.list_tasks(assignee_id=assignee, include_completed=include_done)

    return {
        "ok": True,
        "count": len(tasks),
        "tasks": [
            {"id": t.task_id, "title": t.title, "priority": t.priority.name}
            for t in tasks
        ],
    }
