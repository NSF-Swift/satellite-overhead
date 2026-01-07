from datetime import datetime, timedelta, timezone

import numpy as np

from sopp.models import (
    Configuration,
    Coordinates,
    Facility,
    FrequencyRange,
    Reservation,
    RuntimeSettings,
    Satellite,
    SatelliteTrajectory,
    TimeWindow,
    Position,
)
from sopp.models.ground.config import CustomTrajectoryConfig, StaticPointingConfig
from sopp.models.ground.trajectory import AntennaTrajectory
from sopp.models.satellite import InternationalDesignator, MeanMotion, TleInformation
from sopp.sopp import Sopp
from sopp.utils.time import generate_time_grid
from tests.conftest import ARBITRARY_ALTITUDE, ARBITRARY_AZIMUTH


def assert_trajectories_eq(actual: SatelliteTrajectory, expected: SatelliteTrajectory):
    """
    Helper to compare two trajectories using numpy logic.
    """
    assert actual.satellite.name == expected.satellite.name

    # Compare lengths
    assert len(actual) == len(expected)
    if len(actual) == 0:
        return

    # Compare Arrays (Allow small float tolerance)
    np.testing.assert_array_equal(actual.times, expected.times)
    np.testing.assert_allclose(actual.azimuth, expected.azimuth, atol=1e-5)
    np.testing.assert_allclose(actual.altitude, expected.altitude, atol=1e-5)


class TestSopp:
    def test_full_integration_run(self):
        """
        Runs the full pipeline with real TLEs and real math to verify
        end-to-end connectivity.
        """
        # 1. Setup Data
        res = self._arbitrary_reservation

        # Create a static antenna trajectory pointing at Az=320, Alt=32
        duration = (res.time.end - res.time.begin).total_seconds()
        n_steps = int(duration) + 1  # 1s resolution
        times = np.array(
            [res.time.begin + timedelta(seconds=i) for i in range(n_steps)],
            dtype=object,
        )

        antenna_traj = AntennaTrajectory(
            times=times,
            azimuth=np.full(n_steps, 320.0),
            altitude=np.full(n_steps, 32.0),
        )

        configuration = Configuration(
            reservation=res,
            satellites=self._satellites,
            antenna_config=CustomTrajectoryConfig(antenna_traj),
            runtime_settings=RuntimeSettings(
                time_resolution_seconds=1.0, min_altitude=0.0
            ),
        )

        # 2. Run Sopp
        sopp = Sopp(configuration)

        actual_above_horizon = sopp.get_satellites_above_horizon()
        actual_interference = sopp.get_satellites_crossing_main_beam()

        # 3. Validation

        # Check Satellites Above Horizon
        # Based on TLEs/Time, we expect LILACSAT-2 and NOAA 15 to be visible
        names_visible = {t.satellite.name for t in actual_above_horizon}
        assert "LILACSAT-2" in names_visible
        assert "NOAA 15" in names_visible
        assert "ISS (ZARYA)" not in names_visible  # Should be below horizon/filtered

        # Check Interference
        # LILACSAT-2 is near Az 320 / Alt 32. It should be flagged.
        names_interfering = {t.satellite.name for t in actual_interference}
        assert "LILACSAT-2" in names_interfering

        # NOAA 15 is visible but at Az ~31, Alt ~0. It is FAR from the beam (320, 32).
        assert "NOAA 15" not in names_interfering

    def test_parallel_execution_consistency(self):
        """
        Verifies that running in Parallel Mode (multiprocessing) yields
        identical results to Serial Mode.
        """
        # 1. Setup Data (Use the integration test data)
        res = self._arbitrary_reservation

        # Force a configuration that triggers parallel execution
        # (concurrency > 1 AND enough satellites to trigger chunking)
        # We duplicate the satellites list to simulate a larger load
        many_satellites = (
            self._satellites * 20
        )  # 3 * 20 = 60 sats (triggering chunk logic)

        # 2. Configure for Serial (Control Group)
        config_serial = Configuration(
            reservation=res,
            satellites=many_satellites,
            antenna_config=StaticPointingConfig(Position(320, 32)),
            runtime_settings=RuntimeSettings(concurrency_level=1),
        )

        # 3. Configure for Parallel (Test Group)
        config_parallel = Configuration(
            reservation=res,
            satellites=many_satellites,
            antenna_config=StaticPointingConfig(Position(320, 32)),
            runtime_settings=RuntimeSettings(concurrency_level=2),
        )

        # 4. Execute
        sopp_serial = Sopp(config_serial)
        serial_results = sopp_serial.get_satellites_above_horizon()

        sopp_parallel = Sopp(config_parallel)
        parallel_results = sopp_parallel.get_satellites_above_horizon()

        # 5. Verify
        # Check counts
        assert len(serial_results) > 0
        assert len(serial_results) == len(parallel_results)

        # Check Sort Order (Parallel results might come back out of order)
        # We sort by satellite name + start time to compare
        serial_results.sort(key=lambda x: x.satellite.name)
        parallel_results.sort(key=lambda x: x.satellite.name)

        for s_traj, p_traj in zip(serial_results, parallel_results):
            assert_trajectories_eq(s_traj, p_traj)

    @property
    def _arbitrary_reservation(self) -> Reservation:
        time_window = TimeWindow(
            begin=datetime(2023, 3, 30, 14, 39, 32, tzinfo=timezone.utc),
            end=datetime(2023, 3, 30, 14, 39, 36, tzinfo=timezone.utc),
        )
        return Reservation(
            facility=Facility(
                beamwidth=3.5,
                coordinates=Coordinates(latitude=40.8178049, longitude=-121.4695413),
                name="ARBITRARY_1",
            ),
            time=time_window,
            frequency=FrequencyRange(frequency=135, bandwidth=10),
        )

    @property
    def _satellites(self):
        return [
            self._satellite_in_mainbeam,
            self._satellite_visible_but_safe,
            self._satellite_below_horizon,
        ]

    @property
    def _satellite_in_mainbeam(self) -> Satellite:
        return Satellite(
            name="LILACSAT-2",
            tle_information=TleInformation(
                argument_of_perigee=5.179163326196557,
                drag_coefficient=0.00020184,
                eccentricity=0.0012238,
                epoch_days=26801.52502783,
                inclination=1.7021271170197139,
                international_designator=InternationalDesignator(
                    year=15, launch_number=49, launch_piece="K"
                ),
                mean_anomaly=1.1039888197272412,
                mean_motion=MeanMotion(
                    first_derivative=1.2756659984194665e-10,
                    second_derivative=0.0,
                    value=0.06629635188282393,
                ),
                revolution_number=42329,
                right_ascension_of_ascending_node=2.4726638018364304,
                satellite_number=40908,
                classification="U",
            ),
            frequency=[],
        )

    @property
    def _satellite_visible_but_safe(self) -> Satellite:
        """NOAA 15"""
        return Satellite(
            name="NOAA 15",
            tle_information=TleInformation(
                argument_of_perigee=0.8036979406046088,
                drag_coefficient=0.00010892000000000001,
                eccentricity=0.0011139,
                epoch_days=26801.4833696,
                inclination=1.7210307781480645,
                international_designator=InternationalDesignator(
                    year=98, launch_number=30, launch_piece="A"
                ),
                mean_anomaly=5.4831490673456615,
                mean_motion=MeanMotion(
                    first_derivative=6.6055864051174274e-12,
                    second_derivative=0.0,
                    value=0.06223475712876591,
                ),
                revolution_number=30102,
                right_ascension_of_ascending_node=2.932945522830879,
                satellite_number=25338,
                classification="U",
            ),
            frequency=[],
        )

    @property
    def _satellite_below_horizon(self) -> Satellite:
        """ISS"""
        return Satellite(
            name="ISS (ZARYA)",
            tle_information=TleInformation(
                argument_of_perigee=6.2680236319675116,
                drag_coefficient=0.00018991,
                eccentricity=0.0006492,
                epoch_days=26801.40295236,
                inclination=0.9012601004618398,
                international_designator=InternationalDesignator(
                    year=98, launch_number=67, launch_piece="A"
                ),
                mean_anomaly=0.0168668618912732,
                mean_motion=MeanMotion(
                    first_derivative=3.185528893440345e-10,
                    second_derivative=0.0,
                    value=0.06764422624907401,
                ),
                revolution_number=39717,
                right_ascension_of_ascending_node=2.027426818994173,
                satellite_number=25544,
                classification="U",
            ),
            frequency=[],
        )


def test_reservation_begins_part_way_through_antenna_position_time(
    arbitrary_datetime,
    satellite,
    make_reservation,
    ephemeris_stub,
):
    """
    Scenario: Antenna points at satellite starting at T-1s. Reservation starts at T=0.
    Expected: Should detect interference only starting at T=0.
    """
    # 1. Setup Reservation (Start = T0)
    reservation = make_reservation(start_time=arbitrary_datetime, duration_seconds=2)

    # 2. Setup Mock Antenna Path (Starts at T-1)
    # We create a grid from T-1 to T+2
    t_start_ant = arbitrary_datetime - timedelta(seconds=1)
    t_end_ant = arbitrary_datetime + timedelta(seconds=2)

    times = generate_time_grid(t_start_ant, t_end_ant, resolution_seconds=1)
    n = len(times)

    # Antenna is STATIC, pointing exactly at ARBITRARY_AZIMUTH/ALTITUDE
    trajectory = AntennaTrajectory(
        times=times,
        azimuth=np.full(n, ARBITRARY_AZIMUTH),
        altitude=np.full(n, ARBITRARY_ALTITUDE),
    )

    config = Configuration(
        reservation=reservation,
        satellites=[satellite],
        antenna_config=CustomTrajectoryConfig(trajectory),
    )

    # 3. Setup Sopp
    sopp = Sopp(configuration=config, ephemeris_calculator_class=ephemeris_stub)

    # 4. Execute
    trajectories = sopp.get_satellites_crossing_main_beam()

    # 5. Verify
    assert len(trajectories) == 1
    traj = trajectories[0]

    # Verify the satellite is correct
    assert traj.satellite.name == satellite.name

    # Verify the start time is clipped to the Reservation Start (T0)
    assert traj.times[0] == reservation.time.begin
    assert len(traj) > 0


def test_antenna_positions_that_end_before_reservation_starts_are_not_included(
    arbitrary_datetime,
    satellite,
    make_reservation,
    facility,
    ephemeris_stub,
):
    """
    Scenario: Antenna points at satellite at T-1, but moves away at T=0.
    Expected: No interference found because the overlap happened before the reservation.
    """
    # 1. Setup Reservation (Start = T0)
    reservation = make_reservation(start_time=arbitrary_datetime, duration_seconds=2)

    # 2. Setup Mock Antenna Path
    # Grid: T-1 to T+1
    t_minus_1 = arbitrary_datetime - timedelta(seconds=1)
    t_plus_1 = arbitrary_datetime + timedelta(seconds=1)
    times = generate_time_grid(t_minus_1, t_plus_1, resolution_seconds=1)

    # Define Altitudes
    # T-1: Hits Satellite
    # T=0+: Misses Satellite (Shifted by beamwidth + epsilon)
    alt_hit = ARBITRARY_ALTITUDE
    alt_miss = ARBITRARY_ALTITUDE + facility.beamwidth

    # Create array: [Hit, Miss, Miss]
    altitudes = np.array([alt_hit, alt_miss, alt_miss])
    azimuths = np.full(len(times), ARBITRARY_AZIMUTH)

    trajectory = AntennaTrajectory(times=times, azimuth=azimuths, altitude=altitudes)

    # 3. Setup Sopp
    config = Configuration(
        reservation=reservation,
        satellites=[satellite],
        antenna_config=CustomTrajectoryConfig(trajectory),
    )

    sopp = Sopp(configuration=config, ephemeris_calculator_class=ephemeris_stub)

    # 4. Execute
    trajectories = sopp.get_satellites_crossing_main_beam()

    # 5. Verify
    assert len(trajectories) == 0
