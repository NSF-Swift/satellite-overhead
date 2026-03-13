"""Composable satellite filter chain."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sopp.models.satellite.satellite import Satellite


class Filterer:
    """Applies a chain of filter functions to a satellite list.

    Filters are combined with AND logic: a satellite must pass every
    filter to be included in the output. Supports method chaining.

    Example::

        filterer = (
            Filterer()
            .add_filter(filter_name_contains("STARLINK"))
            .add_filter(filter_orbit_is("leo"))
        )
        filtered = filterer.apply_filters(satellites)
    """

    def __init__(self):
        self._filters: list[Callable[[Satellite], bool]] = []

    def add_filter(self, filter_lambda: Callable[[Satellite], bool]):
        """Add a filter function. Returns self for chaining."""
        if filter_lambda:
            self._filters.append(filter_lambda)
        return self

    def apply_filters(self, elements: list[Satellite]) -> list[Satellite]:
        """Return only satellites that pass all filters."""
        return [
            element for element in elements if all(f(element) for f in self._filters)
        ]
