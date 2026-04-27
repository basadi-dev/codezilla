"""Tests for the CSV parser."""
import pytest
from parser import parse_csv, to_csv


class TestParseCSV:
    def test_simple(self):
        result = parse_csv("a,b,c\n1,2,3")
        assert result == [["a", "b", "c"], ["1", "2", "3"]]

    def test_empty_string(self):
        result = parse_csv("")
        assert result == []

    def test_single_field(self):
        result = parse_csv("hello")
        assert result == [["hello"]]

    def test_quoted_field(self):
        result = parse_csv('"hello, world",foo')
        assert result == [["hello, world", "foo"]]

    def test_escaped_quotes(self):
        result = parse_csv('"say ""hello""",bar')
        assert result == [['say "hello"', "bar"]]

    def test_newline_in_quoted_field(self):
        result = parse_csv('"line1\nline2",other')
        assert result == [["line1\nline2", "other"]]

    def test_whitespace_stripped_from_unquoted(self):
        """Unquoted fields should have leading/trailing whitespace stripped."""
        result = parse_csv("  hello  ,  world  ")
        assert result == [["hello", "world"]]

    def test_whitespace_preserved_in_quoted(self):
        """Quoted fields should preserve whitespace."""
        result = parse_csv('"  hello  ","  world  "')
        assert result == [["  hello  ", "  world  "]]

    def test_trailing_newline(self):
        """A trailing newline should NOT create an extra empty row."""
        result = parse_csv("a,b\n1,2\n")
        assert result == [["a", "b"], ["1", "2"]]

    def test_crlf_line_endings(self):
        """\\r\\n should be treated as a single line ending."""
        result = parse_csv("a,b\r\n1,2\r\n3,4")
        assert result == [["a", "b"], ["1", "2"], ["3", "4"]]

    def test_custom_delimiter(self):
        result = parse_csv("a;b;c\n1;2;3", delimiter=";")
        assert result == [["a", "b", "c"], ["1", "2", "3"]]

    def test_multiple_empty_fields(self):
        result = parse_csv(",,,")
        assert result == [["", "", "", ""]]

    def test_mixed_quoted_unquoted(self):
        result = parse_csv('plain,"quoted",plain2')
        assert result == [["plain", "quoted", "plain2"]]


class TestToCSV:
    def test_simple(self):
        result = to_csv([["a", "b"], ["1", "2"]])
        assert result == "a,b\n1,2"

    def test_quoting_delimiter(self):
        result = to_csv([["hello, world", "foo"]])
        assert result == '"hello, world",foo'

    def test_quoting_newline(self):
        result = to_csv([["line1\nline2", "other"]])
        assert result == '"line1\nline2",other'

    def test_roundtrip(self):
        original = [["name", "bio"], ["Alice", 'She said "hi"'], ["Bob", "line1\nline2"]]
        csv_text = to_csv(original)
        parsed = parse_csv(csv_text)
        assert parsed == original
