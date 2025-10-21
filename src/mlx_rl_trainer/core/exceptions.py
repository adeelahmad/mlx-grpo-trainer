"""
Custom exception classes for the MLX RL Trainer framework.
"""


class CustomBaseException(Exception):
    """Base exception for all custom errors in this project."""

    pass


class ModelLoadError(CustomBaseException):
    """Raised when a model fails to load."""

    pass


class DataLoadError(CustomBaseException):
    """Raised when a dataset fails to load or process."""

    pass


class CheckpointError(CustomBaseException):
    """Raised for errors during checkpoint saving or loading."""

    pass


class InvalidConfigurationError(CustomBaseException):
    """Raised when a configuration is invalid or missing required fields."""

    pass


class TrainingRuntimeError(CustomBaseException):
    """Raised for general errors that occur during the training loop."""

    pass
