"""Built-in satellite filter functions.

Each function returns a predicate ``(Satellite) -> bool`` suitable for
use with :class:`~sopp.filtering.filterer.Filterer`.
"""

import re
from collections.abc import Callable

from sopp.models.core import FrequencyRange
from sopp.models.satellite.satellite import Satellite


def filter_frequency(
    observation_frequency: FrequencyRange,
) -> Callable[[Satellite], bool]:
    """Include satellites whose downlink frequency overlaps the observation band.

    Satellites with no frequency data are included (erring on the side of
    caution). Inactive transmissions are excluded.

    Args:
        observation_frequency: The observation frequency range.
    """

    def filter_function(satellite: Satellite) -> bool:
        if observation_frequency:
            return (
                not satellite.frequency
                or any(sf.frequency is None for sf in satellite.frequency)
                or any(
                    sf.status != "inactive" and observation_frequency.overlaps(sf)
                    for sf in satellite.frequency
                )
            )
        else:
            return True

    return filter_function


def filter_name_regex(regex: str) -> Callable[[Satellite], bool]:
    """Include satellites whose name matches a regex pattern.

    Args:
        regex: Regular expression to match against satellite names.
    """
    pattern = re.compile(regex)

    def filter_function(satellite: Satellite) -> bool:
        return not regex or bool(pattern.search(satellite.name))

    return filter_function


def filter_name_contains(substring: str) -> Callable[[Satellite], bool]:
    """Include satellites whose name contains the given substring.

    Args:
        substring: Substring to search for in satellite names.
    """

    def filter_function(satellite: Satellite) -> bool:
        return not substring or substring in satellite.name

    return filter_function


def filter_name_does_not_contain(substring: str) -> Callable[[Satellite], bool]:
    """Exclude satellites whose name contains the given substring.

    Args:
        substring: Substring to exclude from satellite names.
    """

    def filter_function(satellite: Satellite) -> bool:
        return not substring or not filter_name_contains(substring)(satellite)

    return filter_function


def filter_name_is(substring: str) -> Callable[[Satellite], bool]:
    """Include only satellites whose name matches exactly.

    Args:
        substring: Exact name to match.
    """

    def filter_function(satellite: Satellite) -> bool:
        return not substring or substring == satellite.name

    return filter_function


def filter_orbit_is(orbit_type: str) -> Callable[[Satellite], bool]:
    """Include satellites in a specific orbit regime.

    Args:
        orbit_type: One of 'leo', 'meo', or 'geo'.

    Raises:
        ValueError: If orbit_type is not recognized.
    """

    def filter_function(satellite: Satellite) -> bool:
        if orbit_type == "leo":
            return satellite.orbits_per_day >= 5.0
        elif orbit_type == "meo":
            return satellite.orbits_per_day >= 1.5 and satellite.orbits_per_day < 5.0
        elif orbit_type == "geo":
            return satellite.orbits_per_day >= 0.85 and satellite.orbits_per_day < 1.5
        elif not orbit_type:
            return True
        else:
            raise ValueError("Invalid orbit type. Provide 'leo', 'meo', or 'geo'.")

    return filter_function
