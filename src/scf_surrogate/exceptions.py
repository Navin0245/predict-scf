"""Central exception hierarchy for scf_surrogate.

All exceptions inherit from SCFSurrogateError.

Hierarchy:
    SCFSurrogateError
    +-- DomainError
    |   +-- ParameterRangeError
    |   +-- MissingGapError
    |   +-- PhysicsViolationError
    +-- PipelineError
    |   +-- DataQualityError
    +-- ServingError
        +-- ModelNotLoadedError
"""


class SCFSurrogateError(Exception):
    """Base exception for all scf_surrogate errors."""


class DomainError(SCFSurrogateError):
    """Violation of a physics or engineering domain constraint."""


class ParameterRangeError(DomainError):
    """Input parameter outside DNV-RP-C203 parametric validity range."""


class MissingGapError(DomainError):
    """Gap ratio zeta required for K/KT joints but not provided."""


class PhysicsViolationError(DomainError):
    """Computed result violates a known physical constraint."""


class PipelineError(SCFSurrogateError):
    """Failure within the data or training pipeline."""


class DataQualityError(PipelineError):
    """Input data failed schema validation or quality assertion."""


class ServingError(SCFSurrogateError):
    """Failure within the API serving layer."""


class ModelNotLoadedError(ServingError):
    """Prediction attempted before the model was initialised."""
