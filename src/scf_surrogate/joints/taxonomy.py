"""Joint taxonomy and geometry validation for offshore tubular joints.

Defines the joint type enumeration, per-joint geometric descriptor, and
DNV-RP-C203 parametric bounds validation used throughout the SCF surrogate
data pipeline and serving layer.

Every component in this system that needs to know "what is this joint" imports
from this module. Nothing else in the codebase should define joint types or
DNV validity ranges.

Reference:
    DNV-RP-C203 (2021) Appendix B - Efthymiou SCF parametric equations.
    API RP 2A-WSD (22nd Ed.) Section 4.3.1 - Joint classification.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

from scf_surrogate.exceptions import MissingGapError, ParameterRangeError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DNV-RP-C203 parametric validity ranges
# These are the bounds within which the Efthymiou equations are calibrated.
# Any input outside these ranges is not code-compliant.
# ---------------------------------------------------------------------------

BETA_RANGE: Final[tuple[float, float]] = (0.2, 0.8)
GAMMA_RANGE: Final[tuple[float, float]] = (10.0, 30.0)
TAU_RANGE: Final[tuple[float, float]] = (0.3, 0.8)
THETA_DEG_RANGE: Final[tuple[float, float]] = (30.0, 90.0)
ZETA_RANGE: Final[tuple[float, float]] = (-0.6, 1.0)
ALPHA_RANGE: Final[tuple[float, float]] = (8.0, 40.0)

# ---------------------------------------------------------------------------
# Feature vector column names — must stay in sync with to_feature_vector()
# Defined at module level so pipeline and model code can reference it
# without instantiating a geometry object.
# ---------------------------------------------------------------------------

FEATURE_NAMES: Final[list[str]] = [
    "alpha",
    "beta",
    "gamma",
    "tau",
    "theta_rad",
    "is_TY",
    "is_X",
    "is_KT",
    "is_K",
    "zeta",
]


class JointType(Enum):
    """Offshore tubular joint topology classification.

    Joint type determines which Efthymiou SCF equation set applies.
    The SCF hierarchy is: SCF_X > SCF_Y > SCF_K (DNV-RP-C203 Section B.1).

    Members:
        T_Y: Single brace. T-joint if theta=90 degrees, Y-joint if
            30 <= theta < 90 degrees. Same Efthymiou equation set
            applies to both -- they are not distinguished at the
            equation level, only geometrically.
        X:   Two coaxial braces on opposite chord faces. Load transfers
            THROUGH the chord wall. Highest SCF of all types.
        KT:  Three braces on the same chord face. Two gap values
            required: zeta_AB and zeta_BC. Most complex equation set.
        K:   Two braces on the same chord face with forces that balance
            horizontally in the chord. Requires one gap value zeta.
            Lowest SCF -- chord carries load as shear between footprints.

    Note:
        Classification is load-dependent per API RP 2A Section 4.3.1.
        The same physical joint can be K in one load case and T_Y in
        another. See load_classification.py for load-pattern logic.
    """

    T_Y = auto()
    X = auto()
    KT = auto()
    K = auto()

    @property
    def requires_gap(self) -> bool:
        """Return True if this joint type requires a gap ratio zeta.

        K and KT joints have a defined gap between brace footprints on
        the chord surface. This gap directly enters the Efthymiou SCF
        equations. T_Y and X joints have no such gap parameter.

        Returns:
            bool: True for K and KT joints, False for T_Y and X joints.

        Example:
            >>> JointType.K.requires_gap
            True
            >>> JointType.T_Y.requires_gap
            False
        """
        return self in (JointType.KT, JointType.K)

    @property
    def one_hot_index(self) -> int:
        """Return the 0-based index of this joint type in the one-hot encoding.

        Encoding order follows the JointType enum member definition order:
        T_Y=0, X=1, KT=2, K=3.

        Returns:
            int: Index in range [0, len(JointType) - 1].

        Example:
            >>> JointType.K.one_hot_index
            3
            >>> JointType.T_Y.one_hot_index
            0
        """
        return list(JointType).index(self)

    def to_one_hot(self) -> list[float]:
        """Return a one-hot encoded vector representing this joint type.

        Produces a fixed-length float vector suitable for concatenation
        into the model input feature vector. Exactly one element is 1.0,
        all others are 0.0.

        Returns:
            list[float]: Vector of length 4 with exactly one 1.0 entry.
                Index ordering: [is_TY, is_X, is_KT, is_K].

        Example:
            >>> JointType.K.to_one_hot()
            [0.0, 0.0, 0.0, 1.0]
            >>> JointType.T_Y.to_one_hot()
            [1.0, 0.0, 0.0, 0.0]
            >>> sum(JointType.X.to_one_hot())
            1.0
        """
        # Use float literals so the vector is uniformly list[float].
        # Mixing int and float causes silent type inconsistencies in
        # PyTorch tensors.
        one_hot = [0.0] * len(JointType)
        one_hot[self.one_hot_index] = 1.0
        return one_hot


@dataclass(frozen=True)
class TubularJointGeometry:
    """Immutable geometric descriptor for a single tubular joint.

    Encodes the five Efthymiou parameters plus joint classification into
    a frozen dataclass. Immutability is enforced because a geometry object
    passed into a pipeline must not be mutated by a downstream filter.

    This is the canonical input object for the SCF surrogate pipeline.
    Both the training pipeline and the serving layer use this class as
    their joint representation.

    Attributes:
        alpha: Chord length parameter 2L/D where L is chord half-length
            and D is chord outer diameter. Valid range: [8.0, 40.0].
        beta: Brace-to-chord diameter ratio d/D. Valid range: [0.2, 0.8].
        gamma: Chord slenderness ratio D/2T where T is chord wall
            thickness. Valid range: [10.0, 30.0].
        tau: Brace-to-chord wall thickness ratio t/T. Valid range: [0.3, 0.8].
        theta_deg: Brace inclination angle in degrees measured from the
            chord axis. Valid range: [30.0, 90.0].
        joint_type: Joint topology classification. Determines which
            Efthymiou equation set is applied.
        zeta: Gap ratio g/D where g is the gap between brace footprints
            along the chord surface. Required for K and KT joints.
            Must be None for T_Y and X joints. Valid range: [-0.6, 1.0].
            Negative values indicate overlapping braces.

    Raises:
        MissingGapError: If joint_type is K or KT and zeta is None.

    Example:
        >>> geom = TubularJointGeometry(
        ...     alpha=12.0, beta=0.5, gamma=20.0, tau=0.6,
        ...     theta_deg=45.0, joint_type=JointType.K, zeta=0.1,
        ... )
        >>> geom.theta_rad
        0.7853981633974483
        >>> len(geom.to_feature_vector())
        10
    """

    alpha: float
    beta: float
    gamma: float
    tau: float
    theta_deg: float
    joint_type: JointType
    zeta: float | None = None

    def __post_init__(self) -> None:
        """Validate gap consistency after dataclass initialisation.

        Called automatically by the dataclass machinery after __init__.
        Enforces two rules:
            1. K and KT joints must have zeta provided.
            2. T_Y and X joints should not have zeta provided -- if they
               do, a warning is logged and the value is ignored in all
               downstream calculations.

        Raises:
            MissingGapError: If joint_type requires a gap but zeta is None.
        """
        if self.joint_type.requires_gap and self.zeta is None:
            raise MissingGapError(
                f"Joint type {self.joint_type.name} requires a gap ratio "
                "(zeta) but None was provided."
            )
        if not self.joint_type.requires_gap and self.zeta is not None:
            logger.warning(
                "Gap ratio zeta provided for joint type that does not use it. "
                "Value will be ignored in all downstream calculations.",
                extra={"joint_type": self.joint_type.name, "zeta": self.zeta},
            )

    @property
    def theta_rad(self) -> float:
        """Brace inclination angle in radians.

        Derived from theta_deg. Several Efthymiou equation terms use
        sin(theta) and sin^1.5(theta), so radian form is required.

        Returns:
            float: theta_deg converted to radians.

        Example:
            >>> TubularJointGeometry(
            ...     alpha=12.0, beta=0.5, gamma=20.0, tau=0.6,
            ...     theta_deg=90.0, joint_type=JointType.T_Y
            ... ).theta_rad
            1.5707963267948966
        """
        return math.radians(self.theta_deg)

    def to_feature_vector(self) -> list[float]:
        """Return the fixed-length numeric feature vector for model input.

        Produces a 10-element float vector by concatenating continuous
        parameters with the joint type one-hot encoding and the gap ratio.
        The vector length is always 10 regardless of joint type -- zeta is
        encoded as 0.0 for joint types that do not use it.

        This fixed-length guarantee is required for stacking node feature
        vectors into a PyTorch tensor for GNN training.

        Returns:
            list[float]: Feature vector of length 10 in the order defined
                by FEATURE_NAMES:
                [alpha, beta, gamma, tau, theta_rad,
                 is_TY, is_X, is_KT, is_K, zeta]

        Example:
            >>> geom = TubularJointGeometry(
            ...     alpha=12.0, beta=0.5, gamma=20.0, tau=0.6,
            ...     theta_deg=45.0, joint_type=JointType.T_Y,
            ... )
            >>> geom.to_feature_vector()[-1]   # zeta zero-padded for T_Y
            0.0
            >>> len(geom.to_feature_vector())
            10
        """
        return [
            self.alpha,
            self.beta,
            self.gamma,
            self.tau,
            self.theta_rad,
            *self.joint_type.to_one_hot(),
            # Non-gap joints encode zeta as 0.0 to maintain a fixed-length
            # vector. The joint type one-hot tells the model which joints
            # have a meaningful zeta value.
            self.zeta if self.zeta is not None else 0.0,
        ]


def validate_dnv_bounds(
    geom: TubularJointGeometry,
    strict: bool = True,
) -> list[str]:
    """Validate a joint geometry against DNV-RP-C203 parametric bounds.

    The Efthymiou SCF equations are calibrated against physical test data
    only within the DNV validity ranges. Predictions outside these ranges
    are not code-compliant and must not reach the model.

    Args:
        geom: Joint geometry to validate.
        strict: If True, raises ParameterRangeError on the first violation
            and stops checking. Use True at the API boundary to reject
            invalid requests immediately.
            If False, collects all violations and returns them. Use False
            in the data pipeline to log all issues in one pass.

    Returns:
        list[str]: Human-readable violation messages. Empty list means
            the geometry is fully within DNV parametric bounds.

    Raises:
        ParameterRangeError: If strict=True and any parameter is outside
            its DNV validity range.

    Example:
        >>> geom = TubularJointGeometry(
        ...     alpha=12.0, beta=0.1, gamma=20.0, tau=0.6,
        ...     theta_deg=45.0, joint_type=JointType.T_Y,
        ... )
        >>> validate_dnv_bounds(geom, strict=False)
        ['beta=0.100 outside DNV range [0.2, 0.8]']
    """
    violations: list[str] = []

    checks: list[tuple[str, float, tuple[float, float]]] = [
        ("alpha", geom.alpha, ALPHA_RANGE),
        ("beta", geom.beta, BETA_RANGE),
        ("gamma", geom.gamma, GAMMA_RANGE),
        ("tau", geom.tau, TAU_RANGE),
        ("theta_deg", geom.theta_deg, THETA_DEG_RANGE),
    ]
    if geom.zeta is not None:
        checks.append(("zeta", geom.zeta, ZETA_RANGE))

    for param_name, value, (lo, hi) in checks:
        if not lo <= value <= hi:
            msg = f"{param_name}={value:.3f} outside DNV range [{lo}, {hi}]"
            violations.append(msg)
            if strict:
                raise ParameterRangeError(msg)

    if violations:
        logger.warning(
            "DNV parametric bound violations detected",
            extra={"joint_type": geom.joint_type.name, "violations": violations},
        )
    else:
        logger.debug(
            "DNV bounds check passed",
            extra={
                "joint_type": geom.joint_type.name,
                "beta": geom.beta,
                "gamma": geom.gamma,
                "tau": geom.tau,
                "theta_deg": geom.theta_deg,
            },
        )

    return violations