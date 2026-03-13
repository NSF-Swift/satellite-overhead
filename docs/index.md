# SOPP

**Satellite Orbit Prediction Processor** is an open-source tool for calculating satellite interference to radio astronomy observations.

## Installation

```bash
# Core library only
pip install sopp

# With CLI tools
pip install "sopp[cli]"
```

## Quick Example

```python
from sopp.config.builder import ConfigurationBuilder
from sopp.sopp import Sopp

config = (
    ConfigurationBuilder()
    .set_facility(name="HCRO", latitude=40.8178, longitude=-121.4695, beamwidth=3)
    .set_time_window(begin="2026-01-13T19:00:00", end="2026-01-13T20:00:00")
    .set_frequency_range(frequency=135, bandwidth=10)
    .set_observation_target(declination="40d44m", right_ascension="19h59m")
    .load_satellites(tle_file="satellites.tle")
    .build()
)

engine = Sopp(config)
interference = engine.get_satellites_crossing_main_beam()

for event in interference:
    print(f"{event.satellite.name}: max elev {event.altitude.max():.1f} deg")
```

See the [examples/](https://github.com/NSF-Swift/satellite-overhead/tree/main/examples) directory for more complete usage.

## API Reference

Full API documentation is available in the [Reference](reference/index.md) section.
