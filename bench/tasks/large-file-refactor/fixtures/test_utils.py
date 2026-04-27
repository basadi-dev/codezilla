"""Comprehensive tests for all utility classes."""
import pytest
from utils import Logger, Config, Validator, Formatter


class TestLogger:
    def test_info(self):
        log = Logger()
        log.info("hello")
        assert log.count() == 1
        assert log.entries[0]["message"] == "hello"
        assert log.entries[0]["level"] == "INFO"

    def test_level_filter(self):
        log = Logger(min_level="WARNING")
        log.debug("skip")
        log.info("skip")
        log.warning("keep")
        log.error("keep")
        assert log.count() == 2

    def test_get_entries_by_level(self):
        log = Logger(min_level="DEBUG")
        log.debug("d")
        log.info("i")
        log.error("e")
        assert len(log.get_entries("ERROR")) == 1
        assert len(log.get_entries("DEBUG")) == 1

    def test_clear(self):
        log = Logger()
        log.info("a")
        log.info("b")
        log.clear()
        assert log.count() == 0


class TestConfig:
    def test_get_set(self):
        cfg = Config()
        cfg.set("name", "test")
        assert cfg.get("name") == "test"

    def test_nested(self):
        cfg = Config()
        cfg.set("db.host", "localhost")
        cfg.set("db.port", 5432)
        assert cfg.get("db.host") == "localhost"
        assert cfg.get("db.port") == 5432

    def test_default(self):
        cfg = Config()
        assert cfg.get("missing", "fallback") == "fallback"

    def test_has(self):
        cfg = Config({"x": 1})
        assert cfg.has("x")
        assert not cfg.has("y")

    def test_delete(self):
        cfg = Config({"x": 1, "y": 2})
        assert cfg.delete("x")
        assert not cfg.has("x")
        assert not cfg.delete("z")

    def test_json_roundtrip(self):
        cfg = Config({"a": 1, "b": {"c": 2}})
        json_str = cfg.to_json()
        cfg2 = Config.from_json(json_str)
        assert cfg2.get("a") == 1
        assert cfg2.get("b.c") == 2

    def test_merge(self):
        a = Config({"x": 1, "nested": {"a": 1, "b": 2}})
        b = Config({"y": 2, "nested": {"b": 99, "c": 3}})
        merged = a.merge(b)
        assert merged.get("x") == 1
        assert merged.get("y") == 2
        assert merged.get("nested.a") == 1
        assert merged.get("nested.b") == 99
        assert merged.get("nested.c") == 3

    def test_keys(self):
        cfg = Config({"a": 1, "b": 2, "c": 3})
        assert sorted(cfg.keys()) == ["a", "b", "c"]


class TestValidator:
    def test_required_pass(self):
        v = Validator().required("name")
        ok, errors = v.validate({"name": "Alice"})
        assert ok
        assert errors == []

    def test_required_fail(self):
        v = Validator().required("name")
        ok, errors = v.validate({})
        assert not ok
        assert len(errors) == 1

    def test_min_length(self):
        v = Validator().min_length("password", 8)
        ok, _ = v.validate({"password": "short"})
        assert not ok
        ok, _ = v.validate({"password": "longenough"})
        assert ok

    def test_max_length(self):
        v = Validator().max_length("name", 5)
        ok, _ = v.validate({"name": "toolong"})
        assert not ok

    def test_matches(self):
        v = Validator().matches("email", r"^[^@]+@[^@]+\.[^@]+$")
        ok, _ = v.validate({"email": "test@example.com"})
        assert ok
        ok, _ = v.validate({"email": "invalid"})
        assert not ok

    def test_chained(self):
        v = (Validator()
             .required("name")
             .min_length("name", 2)
             .required("email")
             .matches("email", r"^[^@]+@[^@]+$"))
        ok, errors = v.validate({"name": "A", "email": "bad"})
        assert not ok
        assert len(errors) == 2  # name too short + email no match


class TestFormatter:
    def test_truncate(self):
        assert Formatter.truncate("hello world", 8) == "hello..."
        assert Formatter.truncate("short", 10) == "short"

    def test_slugify(self):
        assert Formatter.slugify("Hello World!") == "hello-world"
        assert Formatter.slugify("  multiple   spaces  ") == "multiple-spaces"

    def test_wrap(self):
        lines = Formatter.wrap("the quick brown fox jumps over the lazy dog", 15)
        assert all(len(line) <= 15 for line in lines)
        assert len(lines) >= 3

    def test_title_case(self):
        assert Formatter.title_case("the lord of the rings") == "The Lord of the Rings"

    def test_indent(self):
        result = Formatter.indent("line1\nline2", spaces=2)
        assert result == "  line1\n  line2"

    def test_indent_skip_first(self):
        result = Formatter.indent("line1\nline2", spaces=2, first_line=False)
        assert result == "line1\n  line2"
