from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

from sopp.models.position_time import PositionTime


class GraphGeneratorPolar:
    """
    The GraphGeneratorPolar class generates a polar graph depicting the Altitude-Azimuth (Alt-Az) position
    of a satellite over time.

    To create a polar graph for a specific satellite's Alt-Az data, instantiate the class with a list of
    PositionTime objects representing the satellite's position at different times. Then, call the
    generate_graph() method to generate and display the polar graph.

    Parameters:
    -----------
    - observation_data (List[PositionTime]): A list of PositionTime objects representing the satellite's
      Alt-Az positions at different times.
    - sat_name: An optional parameter that will be used to create the title of the graph. If not set it
      defaults to 'Satellite'.

    - The polar graph uses a coordinate system where:
      - Azimuth is represented as the theta values.
      - Altitude is represented as the radian values.
      - Azimuth starts at 0 degrees at the top edge of the graph and increases clockwise (or counterclockwise)
        as it goes around the graph.
      - Altitude starts at 0 degrees at the circumference and increases to 90 degrees at the center.

    Example Usage:
    --------------
    # Create an instance of GraphGeneratorPolar with observation data
    generator = GraphGeneratorPolar(observation_data=observation_data, sat_name='ISS')

    # Generate and display the polar graph
    generator.generate_graph()
    """

    def __init__(
        self,
        observation_data: list[PositionTime],
        sat_name: str | None = "Satellite",
    ):
        self._observation_data = observation_data
        self._sat_name = sat_name
        self._label = "Satellite Alt-Az"

    def generate_graph(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")

        self._annotate_start_end_times(ax)
        self._annotate_max_alt(ax)

        ax.plot(self._azimuth_values, self._altitude_values, label=self._label)
        ax.set_title(self._title)

        # setup the polar graph so North is at the top
        # and that altitude increases from edge of graph to center
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(270)
        ax.set_rlim(bottom=90, top=0)
        r_ticks = [0, 20, 40, 60, 80]
        ax.set_rticks(r_ticks)
        ax.set_rgrids(r_ticks[::-1])

        ax.legend()
        plt.tight_layout()
        plt.show()

    def _annotate_start_end_times(self, ax):
        annotate_indices = [0, -1]
        for idx in annotate_indices:
            ax.annotate(
                self._time_values[idx].strftime("%H:%M:%S"),
                (self._azimuth_values[idx], self._altitude_values[idx]),
                ha="center",
                va="top",
            )
            ax.plot(
                self._azimuth_values[idx],
                self._altitude_values[idx],
                marker="+",
                markersize=10,
                color="red",
            )

    def _annotate_max_alt(self, ax):
        idx = np.argmax(self._altitude_values)
        ax.annotate(
            f'Max Alt: {self._altitude_values[idx]:.2f} at {self._time_values[idx].strftime("%H:%M:%S")}',
            (self._azimuth_values[idx], self._altitude_values[idx]),
            ha="center",
            va="top",
        )
        ax.plot(
            self._azimuth_values[idx],
            self._altitude_values[idx],
            marker="+",
            markersize=10,
            color="red",
        )

    @cached_property
    def _title(self):
        start_time = self._time_values[0].strftime("%Y-%m-%d %H:%M:%S")
        end_time = self._time_values[-1].strftime("%H:%M:%S UTC")
        title = f"{self._sat_name} {start_time} - {end_time}"

        return title

    @cached_property
    def _altitude_values(self):
        return np.array([entry.position.altitude for entry in self._observation_data])

    @cached_property
    def _azimuth_values(self):
        return np.radians([entry.position.azimuth for entry in self._observation_data])

    @cached_property
    def _time_values(self):
        return [entry.time for entry in self._observation_data]
