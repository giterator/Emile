"""Emile UI components package.

Components are imported lazily by shell.render() with try/except guards,
so partially-implemented sibling modules (timeline, stream, dashboard, race)
do not break the shell during dev.
"""

__all__ = ["shell"]
