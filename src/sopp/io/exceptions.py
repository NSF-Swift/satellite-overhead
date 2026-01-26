"""Custom exceptions for trajectory I/O operations."""


class TrajectoryIOError(Exception):
    """Base exception for trajectory I/O errors."""

    pass


class TrajectoryFileNotFoundError(TrajectoryIOError):
    """Raised when a trajectory file cannot be found."""

    pass


class TrajectoryFormatError(TrajectoryIOError):
    """Raised when a trajectory file has an invalid format."""

    pass


class TrajectoryValidationError(TrajectoryIOError):
    """Raised when trajectory data fails validation."""

    pass


class TrajectoryMetadataError(TrajectoryIOError):
    """Raised when trajectory metadata is missing or invalid."""

    pass
