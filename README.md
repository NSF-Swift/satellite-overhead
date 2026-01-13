# SOPP - Satellite Orbit Prediction Processor

<div align="left">

| | |
| --- | --- |
| CI/CD | [![CI - Test](https://github.com/NSF-Swift/satellite-overhead/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/NSF-Swift/satellite-overhead/actions/workflows/ci.yml) [![CD - Build](https://github.com/NSF-Swift/satellite-overhead/actions/workflows/cd.yml/badge.svg)](https://github.com/NSF-Swift/satellite-overhead/actions/workflows/cd.yml) |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/sopp.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/sopp/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/sopp.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pypi.org/project/sopp/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sopp.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/sopp/) |

</div>

**SOPP** is a high-performance Python library and CLI tool designed to predict satellite interference for radio astronomy observations.

It uses vectorized orbital mechanics (via Skyfield/SGP4) to simulate thousands of satellites against observation schedules, identifying when a satellite crosses a telescope's main beam or rises above the horizon.

## Installation

Install via pip:

```bash
# Core library only
pip install sopp

# With CLI tools
pip install "sopp[cli]"
```

## CLI Usage

SOPP provides a command-line interface for running simulations and managing data.

### 1. Download Data
First, download the latest TLE (Two-Line Element) data. By default, this pulls active satellites from Celestrak.

```bash
sopp download-tles
```

### 2. Run a Simulation
Run a simulation using a configuration file (see below).

```bash
sopp run --config my_config.json
```

### 3. Ad-Hoc Analysis
You can override configuration parameters directly from the CLI without editing the file.

```bash
# Check for Starlink interference in the next 30 minutes
sopp run --config my_config.json \
    --search STARLINK \
    --orbit leo \
    --start 2025-01-01T12:00:00 \
    --duration 30
```

**Common Options:**
*   `--mode [all|horizon|interference]`: Choose what to calculate.
*   `--limit 10`: Only show the first 10 results.
*   `--local-time`: Display timestamps in your system's local timezone.
*   `--format json`: Output machine-readable JSON.

## Configuration

SOPP uses a JSON configuration file to define the observation parameters.

### Example `config.json`

```json
{
  "facility": {
    "name": "HCRO",
    "latitude": 40.8178049,
    "longitude": -121.4695413,
    "elevation": 986,
    "beamwidth": 3.0
  },
  "frequencyRange": {
    "frequency": 135.0,
    "bandwidth": 10.0
  },
  "reservationWindow": {
    "startTimeUtc": "2026-01-13T12:00:00",
    "endTimeUtc": "2026-01-13T13:00:00"
  },
  "observationTarget": {
    "declination": "-38d6m50.8s",
    "rightAscension": "4h42m"
  },
  "runtimeSettings": {
    "concurrency_level": 4,
    "time_resolution_seconds": 1.0,
    "min_altitude": 0.0
  }
}
```

### Configuration Sections

| Section | Description |
| :--- | :--- |
| **facility** | Defines the ground station location (Lat/Lon/Elev) and antenna beamwidth (degrees). |
| **frequencyRange** | Defines the observation frequency (MHz). Satellites transmitting outside this range are ignored. |
| **reservationWindow** | The start and end time of the observation. |
| **runtimeSettings** | Controls simulation fidelity. `time_resolution_seconds` determines the step size (default 1.0s). |

### Antenna Pointing Modes
You must provide **one** of the following sections to define where the antenna is pointing:

1.  **`observationTarget`**: Tracks a celestial object (RA/Dec).
    ```json
    "observationTarget": { "declination": "...", "rightAscension": "..." }
    ```
2.  **`staticAntennaPosition`**: Points at a fixed Azimuth/Elevation.
    ```json
    "staticAntennaPosition": { "azimuth": 180.0, "altitude": 45.0 }
    ```
3.  **`antennaPositionTimes`**: A custom trajectory (list of time/az/el points).

## Python Library Usage

SOPP is designed to be imported and used directly in Python scripts/projects.

```python
from sopp.config.builder import ConfigurationBuilder
from sopp.filtering.presets import filter_name_does_not_contain
from sopp.sopp import Sopp

# Build Configuration
config = (
    ConfigurationBuilder()
    .set_facility(
        latitude=40.8, longitude=-121.4, elevation=986, name="HCRO", beamwidth=3
    )
    .set_runtime_settings(concurrency_level=4)
    .set_time_window(begin="2026-01-13T19:00:00", end="2026-01-13T20:00:00")
    .set_frequency_range(bandwidth=10, frequency=135)
    # Cygnus A
    .set_observation_target(declination="40d44m", right_ascension="19h59m")
    .load_satellites(tle_file="satellites.tle")
    .add_filter(filter_name_does_not_contain("STARLINK"))
    .build()
)

print(f"Running interference simulation for {len(config.satellites)} satellites:")

# Run Engine
engine = Sopp(config)
interference_events = engine.get_satellites_crossing_main_beam()

# Analyze Results
print(f"Found {len(interference_events)} interference events:")

for event in interference_events:
    start = event.times[0]
    end = event.times[-1]
    duration = (end - start).total_seconds()

    print(f"--- {event.satellite.name} ---")
    print(f"  Window:   {start} -> {end}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Max Elev: {event.altitude.max():.1f} deg")
```

## Data Sources

*   **TLE Data:** Sourced from [Celestrak](https://celestrak.org) (public) or [Space-Track.org](https://www.space-track.org) (requires account/env vars).
*   **Frequency Data:** Optional CSV file to populate satellite transmission frequency. [SSDB](https://github.com/NSF-Swift/sat-frequency-scraper) Format: `ID, Name, Frequency, Bandwidth`.
