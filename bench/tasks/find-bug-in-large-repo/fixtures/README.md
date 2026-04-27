# Task Tracker

A simple task management library with priority-based sorting.

## Structure

- `task_tracker/models/` — Data models (Task, User, Project, Priority)
- `task_tracker/services/` — Business logic
- `task_tracker/storage/` — In-memory persistence
- `task_tracker/utils/` — Utilities (dates, scoring, formatting, validation)
- `task_tracker/api/` — Request handlers and response formatting
- `task_tracker/config/` — Application settings
- `tests/` — Test suite

## Running tests

```bash
python3 -m pytest tests/ -v
```
