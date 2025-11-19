from sopp.models.frequency_range import FrequencyRange
from sopp.models.satellite.satellite import Satellite
from sopp.satellites_filter.filters import filter_frequency


class TestFrequencyFilter:
    def test_single_sat_no_bandwidth(self):
        frequency_filtered_sats = list(
            filter(
                filter_frequency(self._arbitrary_frequency),
                [self._arbitrary_satellite_in_band],
            )
        )
        assert frequency_filtered_sats == [
            Satellite(
                name="name", frequency=[FrequencyRange(frequency=136, bandwidth=None)]
            )
        ]

    def test_two_sats_no_bandwidth(self):
        frequency_filtered_sats = list(
            filter(
                filter_frequency(self._arbitrary_frequency),
                [
                    self._arbitrary_satellite_in_band,
                    self._arbitrary_satellite_out_of_band(),
                ],
            )
        )
        assert frequency_filtered_sats == [
            Satellite(
                name="name", frequency=[FrequencyRange(frequency=136, bandwidth=None)]
            )
        ]

    def test_single_sat_with_bandwidth(self):
        frequency_filtered_sats = list(
            filter(
                filter_frequency(self._arbitrary_frequency),
                [self._arbitrary_satellite_with_bandwidth],
            )
        )
        assert frequency_filtered_sats == [
            Satellite(
                name="name", frequency=[FrequencyRange(frequency=128, bandwidth=10)]
            )
        ]

    def test_inactive_sat(self):
        frequency_filtered_sats = list(
            filter(
                filter_frequency(self._arbitrary_frequency),
                [self._arbitrary_inactive_satellite()],
            )
        )
        assert frequency_filtered_sats == []

    def test_active_and_inactive_sat(self):
        frequency_filtered_sats = list(
            filter(
                filter_frequency(self._arbitrary_frequency),
                [
                    self._arbitrary_satellite_with_bandwidth,
                    self._arbitrary_inactive_satellite(),
                ],
            )
        )
        assert frequency_filtered_sats == [
            Satellite(
                name="name", frequency=[FrequencyRange(frequency=128, bandwidth=10)]
            )
        ]

    def test_no_frequency_data_sat(self):
        no_freq_data_sat = self._arbitrary_satellite_in_band
        no_freq_data_sat.frequency = []

        frequency_filtered_sats = list(
            filter(filter_frequency(self._arbitrary_frequency), [no_freq_data_sat])
        )
        assert frequency_filtered_sats == [Satellite(name="name", frequency=[])]

    def test_frequency_data_none(self):
        frequency_filtered_sats = list(
            filter(
                filter_frequency(self._arbitrary_frequency),
                [self._arbitrary_satellite_freq_is_none],
            )
        )

        assert frequency_filtered_sats == [
            Satellite(
                name="name", frequency=[FrequencyRange(frequency=None, bandwidth=None)]
            )
        ]

    def test_observation_frequency_is_none(self):
        frequency_filtered_sats = list(
            filter(filter_frequency(None), [self._arbitrary_satellite_freq_is_none])
        )

        assert frequency_filtered_sats == [
            Satellite(
                name="name", frequency=[FrequencyRange(frequency=None, bandwidth=None)]
            )
        ]

    @property
    def _arbitrary_satellite_in_band(self) -> Satellite:
        return Satellite(
            name="name", frequency=[FrequencyRange(frequency=136, bandwidth=None)]
        )

    def _arbitrary_satellite_out_of_band(self) -> Satellite:
        return Satellite(
            name="name", frequency=[FrequencyRange(frequency=200, bandwidth=None)]
        )

    @property
    def _arbitrary_satellite_with_bandwidth(self) -> Satellite:
        return Satellite(
            name="name", frequency=[FrequencyRange(frequency=128, bandwidth=10)]
        )

    def _arbitrary_inactive_satellite(self) -> Satellite:
        return Satellite(
            name="name",
            frequency=[FrequencyRange(frequency=130, bandwidth=10, status="inactive")],
        )

    @property
    def _arbitrary_satellite_freq_is_none(self) -> Satellite:
        return Satellite(
            name="name", frequency=[FrequencyRange(frequency=None, bandwidth=None)]
        )

    @property
    def _arbitrary_frequency(self) -> FrequencyRange:
        return FrequencyRange(frequency=135.5, bandwidth=10)
