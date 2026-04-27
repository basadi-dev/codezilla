"""Tests for search service — these should all pass (no bugs here)."""
from datetime import timedelta
from task_tracker.models.priority import Priority


class TestSearchByTitle:
    def test_finds_matching(self, task_service, storage, now):
        from task_tracker.services.search import SearchService
        task_service.create_task("t1", "Deploy hotfix")
        task_service.create_task("t2", "Update docs")
        search = SearchService(storage=storage)
        results = search.search_by_title("hotfix")
        assert len(results) == 1
        assert results[0].task_id == "t1"

    def test_case_insensitive(self, task_service, storage, now):
        from task_tracker.services.search import SearchService
        task_service.create_task("t1", "Deploy HOTFIX")
        search = SearchService(storage=storage)
        results = search.search_by_title("hotfix")
        assert len(results) == 1


class TestSearchByTags:
    def test_finds_by_tag(self, task_service, storage, now):
        from task_tracker.services.search import SearchService
        task_service.create_task("t1", "Bug fix", tags=["bug", "urgent"])
        task_service.create_task("t2", "Feature", tags=["feature"])
        search = SearchService(storage=storage)
        results = search.search_by_tags(["bug"])
        assert len(results) == 1


class TestDueWithinDays:
    def test_finds_upcoming(self, task_service, storage, now):
        from task_tracker.services.search import SearchService
        task_service.create_task(
            "t1", "Soon", deadline=now + timedelta(days=3),
        )
        task_service.create_task(
            "t2", "Far", deadline=now + timedelta(days=30),
        )
        search = SearchService(storage=storage)
        results = search.due_within_days(7, now=now)
        assert len(results) == 1
        assert results[0].task_id == "t1"
