from sopp.custom_dataclasses.coordinates import Coordinates
from sopp.custom_dataclasses.facility import Facility
from sopp.custom_dataclasses.frequency_range.frequency_range import FrequencyRange
from sopp.custom_dataclasses.reservation import Reservation
from sopp.custom_dataclasses.time_window import TimeWindow
from sopp.tardys4_generator import Tardys4Generator
from sopp.utilities import parse_time_and_convert_to_utc


class TestTards4Generator:
    def test_generate(self):
        facility = Facility(
            Coordinates(
                latitude=40.8178049,
                longitude=-121.4695413,
            ),
            elevation=986,
            name="HCRO",
        )

        begin = parse_time_and_convert_to_utc("2024-03-14T09:00:00.000000")
        end = parse_time_and_convert_to_utc("2024-03-14T12:30:00.000000")

        time_window = TimeWindow(
            begin=begin,
            end=end,
        )

        reservation = Reservation(
            facility=facility,
            time=time_window,
            frequency=FrequencyRange(frequency=1575, bandwidth=20),
        )

        generator = Tardys4Generator(
            reservation,
            begin=begin,
            end=end,
        )

        actual = generator.tardys4

        expected = {
            "transactionId": generator._transaction_id,
            "dateTimePublished": generator._time,
            "dateTimeCreated": generator._time,
            "checksum": generator._checksum,
            "scheduledEvents": [
                {
                    "eventId": generator._event_id,
                    "dpaId": None,
                    "locLat": 40.8178049,
                    "locLong": -121.4695413,
                    "locRadius": 1,
                    "locElevation": 986,
                    "coordType": "azel",
                    "eventStatus": "actual",
                    "dateTimeStart": "2024-03-14T09:00:00+00:00",
                    "dateTimeEnd": "2024-03-14T12:30:00+00:00",
                    "freqStart": 1565000000,
                    "freqStop": 1585000000,
                    "regionSize": 3,
                    "regionX": 90,
                    "regionY": 45,
                }
            ],
        }

        assert actual == expected
