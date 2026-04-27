"""Tests for the static file server — including path traversal attack vectors."""
import os
import tempfile
import pytest
from pathlib import Path
from server import FileServer


@pytest.fixture
def server(tmp_path):
    """Create a file server with a temporary document root containing test files."""
    # Create test files
    (tmp_path / "index.html").write_text("<h1>Welcome</h1>")
    (tmp_path / "about.html").write_text("<h1>About</h1>")

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "page.html").write_text("<h1>Sub Page</h1>")
    (sub / "index.html").write_text("<h1>Sub Index</h1>")

    return FileServer(str(tmp_path))


class TestNormalOperation:
    def test_serve_root_file(self, server):
        status, body = server.serve("/index.html")
        assert status == 200
        assert "<h1>Welcome</h1>" in body

    def test_serve_nested_file(self, server):
        status, body = server.serve("/sub/page.html")
        assert status == 200
        assert "Sub Page" in body

    def test_serve_directory_with_index(self, server):
        status, body = server.serve("/sub")
        assert status == 200
        assert "Sub Index" in body

    def test_serve_missing_file(self, server):
        status, body = server.serve("/nonexistent.html")
        assert status == 404

    def test_list_files(self, server):
        status, files = server.list_files("/")
        assert status == 200
        assert "about.html" in files
        assert "index.html" in files

    def test_list_subdir(self, server):
        status, files = server.list_files("/sub")
        assert status == 200
        assert "page.html" in files


class TestPathTraversal:
    """These tests verify the server blocks path traversal attacks."""

    def test_dotdot_traversal(self, server):
        """Basic ../../ attack should be blocked."""
        status, body = server.serve("/../../../etc/passwd")
        assert status == 403, f"Path traversal should return 403, got {status}"

    def test_dotdot_relative(self, server):
        """Relative path traversal should be blocked."""
        status, body = server.serve("/sub/../../etc/passwd")
        assert status == 403

    def test_dotdot_list(self, server):
        """Directory listing outside root should be blocked."""
        status, files = server.list_files("/../..")
        assert status == 403

    def test_encoded_traversal(self, server):
        """URL-encoded traversal should be blocked."""
        status, body = server.serve("/..%2F..%2Fetc/passwd")
        # Even if it resolves to 404, it should not return file contents
        assert status in (403, 404)

    def test_serve_within_root_still_works(self, server):
        """After adding security, normal paths must still work."""
        status, body = server.serve("/about.html")
        assert status == 200
        assert "About" in body

    def test_root_path(self, server):
        """Serving the root itself should work (directory with index)."""
        status, body = server.serve("/")
        # Root directory — might have index.html
        assert status in (200, 403)  # depends on implementation
