"""Tests for task service — including priority sorting.

These tests expose the priority sorting bug: high-priority tasks with
near deadlines should rank above low-priority tasks with distant deadlines.
"""
from datetime import timedelta
from task_tracker.models.priority import Priority


class TestTaskCreation:
    def test_create_task(self, task_service, now):
        task = task_service.create_task("t1", "Test task")
        assert task.task_id == "t1"
        assert task.title == "Test task"

    def test_create_with_priority(self, task_service, now):
        task = task_service.create_task("t1", "Important", priority=Priority.HIGH)
        assert task.priority == Priority.HIGH

    def test_create_with_deadline(self, task_service, now):
        deadline = now + timedelta(days=7)
        task = task_service.create_task("t1", "Deadline task", deadline=deadline)
        assert task.deadline == deadline

    def test_empty_title_raises(self, task_service):
        import pytest
        with pytest.raises(ValueError):
            task_service.create_task("t1", "")


class TestTaskCompletion:
    def test_complete_task(self, task_service):
        task_service.create_task("t1", "Complete me")
        assert task_service.complete_task("t1")
        task = task_service.get_task("t1")
        assert task.completed

    def test_complete_nonexistent(self, task_service):
        assert not task_service.complete_task("nope")


class TestTaskListing:
    def test_list_excludes_completed(self, task_service):
        task_service.create_task("t1", "Active")
        task_service.create_task("t2", "Done")
        task_service.complete_task("t2")
        tasks = task_service.list_tasks()
        assert len(tasks) == 1

    def test_list_includes_completed(self, task_service):
        task_service.create_task("t1", "Active")
        task_service.create_task("t2", "Done")
        task_service.complete_task("t2")
        tasks = task_service.list_tasks(include_completed=True)
        assert len(tasks) == 2

    def test_filter_by_assignee(self, task_service):
        task_service.create_task("t1", "Alice's task", assignee_id="alice")
        task_service.create_task("t2", "Bob's task", assignee_id="bob")
        tasks = task_service.list_tasks(assignee_id="alice")
        assert len(tasks) == 1
        assert tasks[0].assignee_id == "alice"


class TestPrioritySorting:
    """These are the critical tests that expose the sorting bug."""

    def test_urgent_medium_beats_distant_high(self, task_service, now):
        """A MEDIUM task due tomorrow should rank above a HIGH task due in 60 days.

        With correct scoring (multiplication):
            MEDIUM tomorrow: 20 * 10.0 = 200
            HIGH 60 days:    30 * 0.17 = 5.1   → MEDIUM wins ✓

        With the bug (addition):
            MEDIUM tomorrow: 20 + 10.0 = 30
            HIGH 60 days:    30 + 0.17 = 30.17  → HIGH wins ✗
        """
        task_service.create_task(
            "urgent-med", "Fix production alert",
            priority=Priority.MEDIUM,
            deadline=now + timedelta(days=1),
        )
        task_service.create_task(
            "distant-high", "Plan architecture review",
            priority=Priority.HIGH,
            deadline=now + timedelta(days=60),
        )
        sorted_tasks = task_service.sorted_by_priority(now=now)
        assert sorted_tasks[0].task_id == "urgent-med", (
            f"Expected urgent medium task first but got '{sorted_tasks[0].task_id}'. "
            f"Deadline urgency doesn't seem to be affecting the score properly."
        )

    def test_overdue_low_beats_distant_critical(self, task_service, now):
        """An overdue LOW task should beat a CRITICAL task due in 90 days.

        With correct scoring (multiplication):
            LOW overdue:       10 * 20.0 = 200
            CRITICAL 90 days:  40 * 0.11 = 4.4  → LOW-overdue wins ✓

        With the bug (addition):
            LOW overdue:       10 + 20.0 = 30
            CRITICAL 90 days:  40 + 0.11 = 40.11 → CRITICAL wins ✗
        """
        task_service.create_task(
            "overdue-low", "Merge stale branch",
            priority=Priority.LOW,
            deadline=now - timedelta(days=3),  # 3 days overdue
        )
        task_service.create_task(
            "crit-far", "Plan next quarter",
            priority=Priority.CRITICAL,
            deadline=now + timedelta(days=90),
        )
        sorted_tasks = task_service.sorted_by_priority(now=now)
        assert sorted_tasks[0].task_id == "overdue-low", (
            f"Overdue task should rank first but got '{sorted_tasks[0].task_id}'. "
            f"Deadline urgency is not being weighted correctly."
        )

    def test_deadline_urgency_boosts_same_priority(self, task_service, now):
        """Two MEDIUM tasks — the one due sooner should rank higher."""
        task_service.create_task(
            "urgent", "Urgent medium task",
            priority=Priority.MEDIUM,
            deadline=now + timedelta(days=1),
        )
        task_service.create_task(
            "relaxed", "Relaxed medium task",
            priority=Priority.MEDIUM,
            deadline=now + timedelta(days=30),
        )
        sorted_tasks = task_service.sorted_by_priority(now=now)
        assert sorted_tasks[0].task_id == "urgent"

    def test_no_deadline_uses_base_priority(self, task_service, now):
        """Tasks without deadlines should still sort by base priority."""
        task_service.create_task(
            "high", "High no deadline", priority=Priority.HIGH,
        )
        task_service.create_task(
            "low", "Low no deadline", priority=Priority.LOW,
        )
        sorted_tasks = task_service.sorted_by_priority(now=now)
        assert sorted_tasks[0].task_id == "high"

    def test_many_tasks_sorted_correctly(self, task_service, now):
        """Comprehensive sorting test with varied priorities and deadlines."""
        task_service.create_task(
            "crit-soon", "Critical soon",
            priority=Priority.CRITICAL, deadline=now + timedelta(days=1),
        )
        task_service.create_task(
            "med-tomorrow", "Medium tomorrow",
            priority=Priority.MEDIUM, deadline=now + timedelta(days=1),
        )
        task_service.create_task(
            "high-far", "High far away",
            priority=Priority.HIGH, deadline=now + timedelta(days=90),
        )
        task_service.create_task(
            "low-far", "Low far away",
            priority=Priority.LOW, deadline=now + timedelta(days=90),
        )

        sorted_tasks = task_service.sorted_by_priority(now=now)
        ids = [t.task_id for t in sorted_tasks]
        # Critical-soon should be first
        assert ids[0] == "crit-soon"
        # Medium-tomorrow (20*10=200) should beat high-far (30*0.11=3.3)
        assert ids.index("med-tomorrow") < ids.index("high-far"), (
            f"Medium task due tomorrow should rank above high task due in 90 days. "
            f"Got order: {ids}"
        )

