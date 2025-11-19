from sopp.models.configuration import Configuration
from sopp.models.coordinates import Coordinates
from sopp.models.facility import Facility
from sopp.models.frequency_range import FrequencyRange
from sopp.models.reservation import Reservation
from sopp.models.runtime_settings import RuntimeSettings
from sopp.models.satellite.satellite import Satellite
from sopp.models.time_window import TimeWindow


class TestCustomDataclassStr:
    def test_time_window(self):
        expected = (
            "TimeWindow:\n"
            "  Begin:              2024-02-08 10:00:00\n"
            "  End:                2024-02-08 12:00:00"
        )
        assert str(self.time_window) == expected

    def test_frequency_range(self):
        expected = (
            "FrequencyRange:\n"
            "  Frequency:          10 MHz\n"
            "  Bandwidth:          10 MHz"
        )
        assert str(self.frequency_range) == expected

    def test_runtime_settings(self):
        expected = (
            "RuntimeSettings:\n"
            "  Time Interval:      0:00:01\n"
            "  Concurrency:        10"
            "  Min. Altitude:      0.0"
        )
        assert str(self.runtime_settings) == expected

    def test_facility(self):
        expected = (
            "Facility:\n"
            "  Name:               TestFacility\n"
            "  Latitude:           10\n"
            "  Longitude:          10\n"
            "  Elevation:          1000 meters\n"
            "  Beamwidth:          3 degrees"
        )
        assert str(self.facility) == expected

    def test_reservation(self):
        expected = (
            f"Reservation:\n"
            f"{self.facility}\n"
            f"{self.time_window}\n"
            f"{self.frequency_range}"
        )
        assert str(self.reservation) == expected

    def test_configuration(self):
        expected = (
            f"Configuration:\n"
            f"{self.reservation}\n"
            f"{self.runtime_settings}\n"
            f"Satellites:           1 total"
        )
        assert str(self.configuration) == expected

    @property
    def time_window(self):
        return TimeWindow(begin="2024-02-08 10:00:00", end="2024-02-08 12:00:00")

    @property
    def frequency_range(self):
        return FrequencyRange(10, 10)

    @property
    def runtime_settings(self):
        return RuntimeSettings(1, 10)

    @property
    def facility(self):
        return Facility(Coordinates(10, 10), 3, 1000, "TestFacility")

    @property
    def reservation(self):
        return Reservation(self.facility, self.time_window, self.frequency_range)

    @property
    def configuration(self):
        return Configuration(
            self.reservation, [Satellite(name="Test")], [], self.runtime_settings
        )
