import json
import time
from pathlib import Path
from typing import Annotated, Optional
from enum import Enum
from datetime import datetime, timedelta, timezone
from contextlib import nullcontext

import typer
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from sopp.config.builder import ConfigurationBuilder
from sopp.filtering.presets import (
    filter_frequency,
    filter_name_contains,
    filter_orbit_is,
    filter_name_does_not_contain,
)

from sopp.io.tle import fetch_tles
from sopp.sopp import Sopp
from sopp.models.satellite.trajectory import SatelliteTrajectory
from sopp.__about__ import __version__ as APP_VERSION


# Enums
class AnalysisMode(str, Enum):
    all = "all"
    horizon = "horizon"
    interference = "interference"


class OrbitType(str, Enum):
    leo = "leo"
    meo = "meo"
    geo = "geo"


class OutputFormat(str, Enum):
    table = "table"
    json = "json"


app = typer.Typer(
    name="sopp",
    help="Satellite Orbit Preprocessor - Radio Astronomy Interference Analysis",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to the JSON configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    tle_file: Annotated[
        Path,
        typer.Option(
            "--tle",
            "-t",
            help="Path to TLE file. Downloads from Celestrak if missing.",
        ),
    ] = Path("satellites.tle"),
    frequency_file: Annotated[
        Optional[Path],
        typer.Option(
            "--freq",
            "-f",
            help="Path to CSV frequency data file.",
            exists=True,
            file_okay=True,
        ),
    ] = None,
    # Analysis Options
    mode: Annotated[
        AnalysisMode,
        typer.Option(
            "--mode", "-m", help="Analysis mode: horizon, interference, or all."
        ),
    ] = AnalysisMode.all,
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", "-n", help="Limit number of rows displayed in output."),
    ] = None,
    local_time: Annotated[
        bool,
        typer.Option(
            "--local-time",
            help="Display times in local system timezone instead of UTC.",
        ),
    ] = False,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format", help="Output format: table (human) or json (machine)."
        ),
    ] = OutputFormat.table,
    # Time Overrides
    start_time: Annotated[
        Optional[datetime],
        typer.Option(
            "--start",
            help="Override start time (ISO format, e.g. '2025-12-15T12:00:00'). Assumes UTC.",
        ),
    ] = None,
    end_time: Annotated[
        Optional[datetime],
        typer.Option(
            "--end",
            help="Override end time (ISO format, e.g. '2025-12-15T13:00:00'). Assumes UTC.",
        ),
    ] = None,
    duration_minutes: Annotated[
        Optional[float],
        typer.Option(
            "--duration",
            help="Override duration in minutes (used with --start to calculate end time).",
        ),
    ] = None,
    # Filters
    search: Annotated[
        Optional[str],
        typer.Option("--search", help="Filter satellites by name (substring match)."),
    ] = None,
    exclude: Annotated[
        Optional[list[str]],
        typer.Option(
            "--exclude", help="Exclude satellites (can be used multiple times)."
        ),
    ] = None,
    orbit_type: Annotated[
        Optional[OrbitType],
        typer.Option("--orbit", help="Filter by orbit type (LEO/MEO/GEO)."),
    ] = None,
    # Altitude Override
    override_min_alt: Annotated[
        Optional[float],
        typer.Option(
            "--min-alt", help="Override minimum altitude (degrees) from config."
        ),
    ] = None,
):
    """
    Run a simulation based on a configuration file.
    """
    if output_format == OutputFormat.table:
        _print_banner()

    # TLE Management
    if not tle_file.exists():
        if output_format == OutputFormat.table:
            console.print(
                f"[yellow]TLE file not found at {tle_file}. Downloading...[/yellow]"
            )
        try:
            fetch_tles(output_path=tle_file, source="celestrak")
        except Exception as e:
            console.print(f"[bold red]Failed to download TLEs:[/bold red] {e}")
            raise typer.Exit(code=1)

    # Build Configuration
    try:
        status_ctx = (
            console.status("[bold green]Building Configuration...")
            if output_format == OutputFormat.table
            else nullcontext()
        )

        with status_ctx:
            builder = ConfigurationBuilder()
            builder.set_from_config_file(config_file=config)
            builder.load_satellites(tle_file=tle_file, frequency_file=frequency_file)

            # Apply Filters
            if builder.frequency_range:
                builder.add_filter(filter_frequency(builder.frequency_range))

            if search:
                builder.add_filter(filter_name_contains(search))

            if exclude:
                for ex in exclude:
                    builder.add_filter(filter_name_does_not_contain(ex))

            if orbit_type:
                builder.add_filter(filter_orbit_is(orbit_type.value))

            # Apply Settings Overrides
            if override_min_alt is not None:
                builder.runtime_settings.min_altitude = override_min_alt

            # Apply Time Overrides
            if start_time or end_time or duration_minutes:
                # Fallback to what was in the config file
                current_start = (
                    builder.time_window.begin if builder.time_window else None
                )
                current_end = builder.time_window.end if builder.time_window else None

                # Determine Start
                new_start = start_time if start_time else current_start
                # Enforce UTC
                if new_start and new_start.tzinfo is None:
                    new_start = new_start.replace(tzinfo=timezone.utc)

                # Determine End
                new_end = end_time if end_time else current_end

                # Calculate end from duration if provided
                if duration_minutes and new_start:
                    new_end = new_start + timedelta(minutes=duration_minutes)
                elif new_end and new_end.tzinfo is None:
                    new_end = new_end.replace(tzinfo=timezone.utc)

                if new_start and new_end:
                    builder.set_time_window(new_start, new_end)
                else:
                    console.print(
                        "[yellow]Warning: Time override ignored. "
                        "You must provide enough info to determine both Start and End "
                        "(e.g., --start and --duration, or --start and --end).[/yellow]"
                    )

            configuration = builder.build()

    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    sopp_engine = Sopp(configuration)

    satellites_above_horizon = []
    interference_windows = []

    t0 = time.perf_counter()

    if output_format == OutputFormat.table:
        _print_summary(configuration, len(configuration.satellites))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if mode in (AnalysisMode.all, AnalysisMode.horizon):
                task1 = progress.add_task(
                    "Calculating Horizon Visibility...", total=None
                )
                satellites_above_horizon = sopp_engine.get_satellites_above_horizon()
                progress.update(task1, completed=True)

            if mode in (AnalysisMode.all, AnalysisMode.interference):
                task2 = progress.add_task(
                    "Calculating Beam Interference...", total=None
                )
                interference_windows = sopp_engine.get_satellites_crossing_main_beam()
                progress.update(task2, completed=True)
    else:
        # Silent mode
        if mode in (AnalysisMode.all, AnalysisMode.horizon):
            satellites_above_horizon = sopp_engine.get_satellites_above_horizon()
        if mode in (AnalysisMode.all, AnalysisMode.interference):
            interference_windows = sopp_engine.get_satellites_crossing_main_beam()

    # End Timer
    elapsed_seconds = time.perf_counter() - t0

    # Output
    if output_format == OutputFormat.json:
        _print_json(satellites_above_horizon, interference_windows)
    else:
        _print_results_header(
            mode, satellites_above_horizon, interference_windows, elapsed_seconds
        )

        if (
            mode in (AnalysisMode.all, AnalysisMode.horizon)
            and satellites_above_horizon
        ):
            _print_horizon_table(satellites_above_horizon, limit, local_time)

        if (
            mode in (AnalysisMode.all, AnalysisMode.interference)
            and interference_windows
        ):
            _print_interference_table(interference_windows, limit, local_time)


@app.command()
def download_tles(
    output: Annotated[Path, typer.Argument(help="Path to save TLE file")] = Path(
        "satellites.tle"
    ),
    source: Annotated[
        str, typer.Option(help="Source to fetch from (celestrak/spacetrack)")
    ] = "celestrak",
):
    try:
        path = fetch_tles(output_path=output, source=source)
        console.print(f"[green]Successfully saved TLEs to {path}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error fetching TLEs:[/bold red] {e}")
        raise typer.Exit(code=1)


# Helpers


def _format_dt(dt: datetime, use_local: bool) -> str:
    if use_local:
        return dt.astimezone(None).strftime("%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _print_banner():
    console.print(
        Panel.fit(
            "[bold magenta]SOPP[/bold magenta]\nSatellite Orbit Preprocessor",
            subtitle=APP_VERSION,
            border_style="magenta",
        )
    )


def _print_summary(configuration, sat_count):
    res = configuration.reservation
    console.print(
        Panel(
            f"[bold]Facility:[/bold] {res.facility.name}\n"
            f"[bold]Time:[/bold] {res.time.begin} -> {res.time.end}\n"
            f"[bold]Frequency:[/bold] {res.frequency}\n"
            f"[bold]Satellites Loaded:[/bold] {sat_count}",
            title="Simulation Parameters",
            border_style="blue",
        )
    )


def _print_results_header(mode, horizon_sats, interference_sats, elapsed):
    """
    Prints the results summary dynamically based on mode.
    """
    console.print("\n[bold underline]Results[/bold underline]")

    # Only print lines relevant to the requested mode
    if mode in (AnalysisMode.all, AnalysisMode.horizon):
        console.print(f"Satellites above horizon: [cyan]{len(horizon_sats)}[/cyan]")

    if mode in (AnalysisMode.all, AnalysisMode.interference):
        console.print(
            f"Main Beam crossings:      [bold red]{len(interference_sats)}[/bold red]"
        )

    console.print(
        f"Computation Time:         [bold green]{elapsed:.2f} seconds[/bold green]\n"
    )


def _print_horizon_table(
    windows: list[SatelliteTrajectory], limit: int | None, local_time: bool
):
    table = Table(title="Satellites Above Horizon")
    table.add_column("#", style="dim")
    table.add_column("Satellite", style="cyan")
    table.add_column("Rise Time", style="green")
    table.add_column("Set Time", style="yellow")
    table.add_column("Max Alt", justify="right", style="magenta")
    table.add_column("Time of Max", justify="right", style="blue")

    count = 0
    for i, window in enumerate(windows, start=1):
        if len(window) == 0:
            continue

        max_idx = np.argmax(window.altitude)
        max_alt = window.altitude[max_idx]
        max_time = window.times[max_idx]

        table.add_row(
            str(i),
            window.satellite.name,
            _format_dt(window.times[0], local_time),
            _format_dt(window.times[-1], local_time),
            f"{max_alt:.1f}°",
            _format_dt(max_time, local_time),
        )
        count += 1
        if limit and count >= limit:
            break

    if limit and len(windows) > limit:
        table.caption = f"... and {len(windows) - limit} more"

    console.print(table)
    console.print()


def _print_interference_table(
    windows: list[SatelliteTrajectory], limit: int | None, local_time: bool
):
    table = Table(title="Main Beam Interference Events", style="red")
    table.add_column("#", style="dim")
    table.add_column("Satellite", style="cyan")
    table.add_column("Start", style="green")
    table.add_column("End", style="red")
    table.add_column("Duration (s)", justify="right")

    count = 0
    for i, window in enumerate(windows, start=1):
        if len(window) == 0:
            continue

        duration = (window.times[-1] - window.times[0]).total_seconds()

        table.add_row(
            str(i),
            window.satellite.name,
            _format_dt(window.times[0], local_time),
            _format_dt(window.times[-1], local_time),
            f"{duration:.1f}",
        )
        count += 1
        if limit and count >= limit:
            break

    if limit and len(windows) > limit:
        table.caption = f"... and {len(windows) - limit} more"

    console.print(table)


def _print_json(horizon, interference):
    def serialize(traj_list):
        data = []
        for t in traj_list:
            if len(t) == 0:
                continue
            data.append(
                {
                    "satellite": t.satellite.name,
                    "satellite_id": t.satellite.tle_information.satellite_number,
                    "start": t.times[0].isoformat(),
                    "end": t.times[-1].isoformat(),
                    "duration_sec": (t.times[-1] - t.times[0]).total_seconds(),
                    "max_altitude_deg": float(np.max(t.altitude)),
                }
            )
        return data

    output = {
        "above_horizon": serialize(horizon),
        "interference": serialize(interference),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
