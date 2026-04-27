"""Priority scoring — computes a composite score for task ranking.

The score combines the task's base priority level with an urgency factor
derived from how close the deadline is. Tasks with no deadline use a
default urgency of 1.0.

Score formula:
    score = base_priority_weight * urgency_multiplier

Where:
    base_priority_weight = priority.value * 10
    urgency_multiplier   = max(1.0, 10.0 / max(days_until_deadline, 0.1))

This means a CRITICAL task (value=4) due tomorrow gets:
    4 * 10 * (10.0 / 1.0) = 400

While a LOW task (value=1) due in 30 days gets:
    1 * 10 * (10.0 / 30.0) = 3.3
"""
from datetime import datetime
from typing import Optional

from ..models.task import Task


def compute_priority_score(task: Task, now: Optional[datetime] = None) -> float:
    """Compute a numeric priority score for ranking.

    Higher score = higher priority. Combines the task's priority level
    with deadline urgency so that imminent tasks rank higher.
    """
    ref = now or datetime.now()
    base_weight = task.priority.value * 10.0
    urgency = _deadline_urgency(task, ref)

    # BUG: should be base_weight * urgency (multiplication),
    # but uses addition instead. This means deadline urgency barely
    # affects the score — a CRITICAL task with no deadline (urgency=1.0)
    # gets 40+1=41, while a LOW task due tomorrow (urgency=10.0) gets
    # 10+10=20. The critical task always wins regardless of deadline.
    return base_weight + urgency


def _deadline_urgency(task: Task, now: datetime) -> float:
    """Compute urgency multiplier from deadline proximity.

    Returns a value >= 1.0. Closer deadlines produce higher urgency.
    Tasks with no deadline return 1.0 (neutral).
    """
    if task.deadline is None:
        return 1.0

    days_left = task.days_until_deadline(now)
    if days_left is None:
        return 1.0

    if days_left <= 0:
        # Overdue — maximum urgency
        return 20.0

    # Urgency increases as deadline approaches
    # At 10 days out → 1.0, at 1 day out → 10.0, at 0.1 days → 100.0
    return max(1.0, 10.0 / max(days_left, 0.1))
