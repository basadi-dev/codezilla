"""A simple stack data structure."""


class StackEmpty(Exception):
    """Raised when trying to pop or peek from an empty stack."""
    pass


class Stack:
    """A LIFO stack implemented with a Python list."""

    def __init__(self):
        self._items: list = []

    def push(self, item) -> None:
        """Push an item onto the top of the stack."""
        self._items.append(item)

    def pop(self):
        """Remove and return the top item. Raises StackEmpty if empty."""
        if self.is_empty():
            raise StackEmpty("cannot pop from an empty stack")
        return self._items.pop()

    def peek(self):
        """Return the top item without removing it. Raises StackEmpty if empty."""
        if self.is_empty():
            raise StackEmpty("cannot peek at an empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        """Return True if the stack has no items."""
        return len(self._items) == 0

    def size(self) -> int:
        """Return the number of items in the stack."""
        return len(self._items)

    def clear(self) -> None:
        """Remove all items from the stack."""
        self._items.clear()

    def __contains__(self, item) -> bool:
        """Support 'in' operator: check if item exists in the stack."""
        return item in self._items

    def __iter__(self):
        """Iterate top-to-bottom (most recently pushed first)."""
        return reversed(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        items_str = ", ".join(repr(x) for x in reversed(self._items))
        return f"Stack([{items_str}])"
