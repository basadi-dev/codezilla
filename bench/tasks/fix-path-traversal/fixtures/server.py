"""Simple static file server with a path traversal vulnerability."""
import os
from pathlib import Path


class FileServer:
    """Serves files from a document root directory."""

    def __init__(self, document_root: str):
        self.document_root = Path(document_root).resolve()

    def serve(self, request_path: str) -> tuple[int, str]:
        """Serve a file given a request path.

        Returns (status_code, body):
        - (200, file_contents) for valid files
        - (403, "Forbidden") for paths outside the root
        - (404, "Not Found") for missing files
        """
        # VULNERABLE: does not validate that the resolved path is within document_root
        # An attacker can use ../../ to escape the root directory
        file_path = self.document_root / request_path.lstrip("/")

        if not file_path.exists():
            return (404, "Not Found")

        if file_path.is_dir():
            index = file_path / "index.html"
            if index.exists():
                return (200, index.read_text())
            return (403, "Forbidden")

        return (200, file_path.read_text())

    def list_files(self, request_path: str = "/") -> tuple[int, list[str]]:
        """List files in a directory.

        Returns (status_code, file_list):
        - (200, [...]) for valid directories
        - (403, []) for paths outside the root
        - (404, []) for missing directories
        """
        dir_path = self.document_root / request_path.lstrip("/")

        if not dir_path.exists():
            return (404, [])

        if not dir_path.is_dir():
            return (400, [])

        files = sorted(str(p.relative_to(dir_path)) for p in dir_path.iterdir())
        return (200, files)
