def test_satellite_always_above_horizon_is_returned(
    make_event_finder, satellite, reservation
):
    """
    Scenario: Satellite is visible for the entire window.
    Expected: One trajectory returned covering the full duration.
    """
    finder = make_event_finder(reservation=reservation, satellites=[satellite])

    trajectories = finder.get_satellites_above_horizon()

    assert len(trajectories) == 1
    assert trajectories[0].satellite.name == satellite.name
    # Verify we got data points
    assert len(trajectories[0]) > 0
