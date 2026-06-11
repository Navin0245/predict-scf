from __future__ import annotations

import math

import pytest

from scf_surrogate.exceptions import (
    MissingGapError,
    ParameterRangeError,
)
from scf_surrogate.joints.taxonomy import (
    ALPHA_RANGE,
    BETA_RANGE,
    FEATURE_NAMES,
    GAMMA_RANGE,
    TAU_RANGE,
    THETA_DEG_RANGE,
    JointType,
    TubularJointGeometry,
    validate_dnv_bounds,
)


@pytest.fixture()
def valid_ty_geom() -> TubularJointGeometry:
    """Return a valid T/Y joint geometry fixture."""
    return TubularJointGeometry(
        alpha= 8.0, beta=0.5, gamma=20.0, tau=0.6, theta_deg=60.0, joint_type=JointType.T_Y
    )
 
 
@pytest.fixture()
def valid_k_geom() -> TubularJointGeometry:
    """Return a valid K joint geometry fixture with a required gap ratio."""
    return TubularJointGeometry(
        alpha= 8.0, beta=0.5, gamma=20.0, tau=0.6, theta_deg=45.0,
        joint_type=JointType.K, zeta=0.1
    )
 
 
@pytest.fixture()
def valid_x_geom() -> TubularJointGeometry:
    """Return a valid X joint geometry fixture."""
    return TubularJointGeometry(
        alpha= 8.0, beta=0.4, gamma=15.0, tau=0.5, theta_deg=90.0, joint_type=JointType.X
    )


class TestJointTypeEnum:
    """Invariant: one-hot vectors must be mutually exclusive and exhaustive."""
 
    def test_one_hot_ty(self) -> None:
        """Verify that T/Y joints encode to the first one-hot position."""
        assert JointType.T_Y.to_one_hot() == [1.0, 0.0, 0.0, 0.0]
 
    def test_one_hot_x(self) -> None:
        """Verify that X joints encode to the second one-hot position."""
        assert JointType.X.to_one_hot() == [0.0, 1.0, 0.0, 0.0]
 
    def test_one_hot_k(self) -> None:
        """Verify that K joints encode to the fourth one-hot position."""
        assert JointType.K.to_one_hot() == [0.0, 0.0, 0.0, 1.0]
 
    def test_one_hot_kt(self) -> None:
        """Verify that KT joints encode to the third one-hot position."""
        assert JointType.KT.to_one_hot() == [0.0, 0.0, 1.0, 0.0]

    def test_one_hot_index_ordering(self) -> None:
        """Index ordering must match FEATURE_NAMES: T_Y=0, X=1, KT=2, K=3."""
        assert JointType.T_Y.one_hot_index == 0
        assert JointType.X.one_hot_index == 1
        assert JointType.KT.one_hot_index == 2
        assert JointType.K.one_hot_index == 3

    def test_one_hot_returns_float_not_int(self) -> None:
        """Verify one-hot vectors contain floats for PyTorch tensor consistency."""
        for jt in JointType:
            vec = jt.to_one_hot()
            for element in vec:
                assert isinstance(element, float), (
                    f"{jt.name}.to_one_hot() returned int, expected float"
                )
                
    def test_one_hot_sum_is_one(self) -> None:
        """Verify each joint type has exactly one active one-hot value."""
        for jt in JointType:
            assert sum(jt.to_one_hot()) == 1.0, f"One-hot sum != 1 for {jt.name}"
 
    def test_requires_gap_k_kt(self) -> None:
        """Verify K and KT joints require a gap ratio."""
        assert JointType.K.requires_gap is True
        assert JointType.KT.requires_gap is True
 
    def test_requires_gap_ty_x(self) -> None:
        """Verify T/Y and X joints do not require a gap ratio."""
        assert JointType.T_Y.requires_gap is False
        assert JointType.X.requires_gap is False



class TestGeometryConstruction:
 
    def test_theta_rad_conversion(self, valid_ty_geom: TubularJointGeometry) -> None:
        """Invariant: theta_rad = radians(theta_deg) to float precision."""
        assert math.isclose(valid_ty_geom.theta_rad, math.radians(60.0))
 
    def test_k_joint_without_zeta_raises(self) -> None:
        """K-joint missing gap ratio must raise MissingGapError, not silently proceed."""
        with pytest.raises(MissingGapError):
            TubularJointGeometry(
                alpha= 8.0, beta=0.5, gamma=20.0, tau=0.6,
                theta_deg=45.0, joint_type=JointType.K,
                zeta=None,  # missing!
            )
 
    def test_kt_joint_without_zeta_raises(self) -> None:
        """Verify KT joints raise when the required gap ratio is missing."""
        with pytest.raises(MissingGapError):
            TubularJointGeometry(
                alpha= 8.0, beta=0.5, gamma=20.0, tau=0.6,
                theta_deg=45.0, joint_type=JointType.KT,
            )
 
    def test_ty_joint_with_zeta_does_not_raise(self) -> None:
        """T/Y joint with gap provided should warn but not raise."""
        geom = TubularJointGeometry(
            alpha= 8.0, beta=0.5, gamma=20.0, tau=0.6,
            theta_deg=60.0, joint_type=JointType.T_Y,
            zeta=0.1,  # irrelevant but not an error
        )
        assert geom is not None
 
    def test_immutability(self, valid_ty_geom: TubularJointGeometry) -> None:
        """Frozen dataclass must reject attribute mutation."""
        with pytest.raises(Exception):  # FrozenInstanceError (AttributeError in Python <3.11)
            valid_ty_geom.beta = 0.9  # type: ignore[misc]


 
class TestFeatureVector:
 
    def test_feature_vector_length(self, valid_k_geom: TubularJointGeometry) -> None:
        """Feature vector must be exactly 10 elements (6 continuous + 4 one-hot)."""
        assert len(valid_k_geom.to_feature_vector()) == 10
 
    def test_feature_vector_k_joint(self, valid_k_geom: TubularJointGeometry) -> None:
        """Feature vector elements must match expected values within float precision."""
        vec = valid_k_geom.to_feature_vector()
        expected = [8.0, 0.5, 20.0, 0.6, math.radians(45.0), 0.0, 0.0, 0.0, 1.0, 0.1]
        assert len(vec) == len(expected)
        for actual, exp in zip(vec, expected):
            assert math.isclose(actual, exp, rel_tol=1e-9), (
                f"Expected {exp}, got {actual}"
            )
 
    def test_feature_vector_ty_joint(self, valid_ty_geom: TubularJointGeometry) -> None:
        """Verify T/Y joint feature vectors include expected values and padding."""
        vec = valid_ty_geom.to_feature_vector()
        expected = [8.0, 0.5, 20.0, 0.6, math.radians(60.0), 1.0, 0.0, 0.0, 0.0, 0.0]
        assert len(vec) == len(expected)
        for actual, exp in zip(vec, expected):
            assert math.isclose(actual, exp, rel_tol=1e-9), (
                f"Expected {exp}, got {actual}"
            )

    def test_ty_joint_zeta_is_zero_padded(self, valid_ty_geom: TubularJointGeometry) -> None:
        """T_Y joint must encode zeta as 0.0 to maintain fixed-length vector."""
        vec = valid_ty_geom.to_feature_vector()
        assert vec[-1] == 0.0, "zeta must be 0.0 for T_Y joints, not None or missing"

    def test_x_joint_zeta_is_zero_padded(self, valid_x_geom: TubularJointGeometry) -> None:
        """X joint must encode zeta as 0.0 to maintain fixed-length vector."""
        vec = valid_x_geom.to_feature_vector()
        assert vec[-1] == 0.0, "zeta must be 0.0 for X joints, not None or missing"

    def test_feature_names_length(self) -> None:
        """Verify feature names cover all ten feature vector positions."""
        assert len(FEATURE_NAMES) == 10
 
    def test_feature_names_match_vector(self, valid_ty_geom: TubularJointGeometry) -> None:
        """Verify feature names stay aligned with the generated feature vector."""
        names = FEATURE_NAMES
        vec = valid_ty_geom.to_feature_vector()
        assert len(names) == len(vec)

 
class TestDNVValidation:
 
    def test_valid_geometry_no_violations(self, valid_ty_geom: TubularJointGeometry) -> None:
        """Verify valid geometry produces no DNV bound violations."""
        violations = validate_dnv_bounds(valid_ty_geom, strict=False)
        assert violations == []
 
    def test_beta_below_range_strict(self) -> None:
        """Verify strict validation raises when beta is below the DNV range."""
        geom = TubularJointGeometry(
            alpha= 8.0, beta=0.1,  # below 0.2
            gamma=20.0, tau=0.6, theta_deg=60.0, joint_type=JointType.T_Y
        )
        with pytest.raises(ParameterRangeError, match="beta"):
            validate_dnv_bounds(geom, strict=True)

    def test_alpha_below_range_non_strict(self) -> None:
        """Verify non-strict validation reports alpha below the DNV range."""
        geom = TubularJointGeometry(
            alpha=5.0,   # below 8.0
            beta=0.5, gamma=20.0, tau=0.6,
            theta_deg=60.0, joint_type=JointType.T_Y
        )
        violations = validate_dnv_bounds(geom, strict=False)
        assert any("alpha" in v for v in violations)

    def test_tau_above_range_non_strict(self) -> None:
        """Verify non-strict validation reports tau above the DNV range."""
        geom = TubularJointGeometry(
            alpha=12.0, beta=0.5, gamma=20.0,
            tau=0.95,   # above 0.8
            theta_deg=60.0, joint_type=JointType.T_Y
        )
        violations = validate_dnv_bounds(geom, strict=False)
        assert any("tau" in v for v in violations)

    def test_gamma_above_range_non_strict(self) -> None:
        """Verify non-strict validation reports gamma above the DNV range."""
        geom = TubularJointGeometry(
            alpha= 8.0, beta=0.5, gamma=35.0,  # above 30.0
            tau=0.6, theta_deg=60.0, joint_type=JointType.T_Y
        )
        violations = validate_dnv_bounds(geom, strict=False)
        assert any("gamma" in v for v in violations)
 
    def test_theta_below_range_non_strict(self) -> None:
        """Verify non-strict validation reports theta below the DNV range."""
        geom = TubularJointGeometry(
            alpha= 8.0, beta=0.5, gamma=20.0, tau=0.6,
            theta_deg=20.0,  # below 30°
            joint_type=JointType.T_Y
        )
        violations = validate_dnv_bounds(geom, strict=False)
        assert any("theta_deg" in v for v in violations)
 
    def test_zeta_out_of_range_k_joint_strict(self) -> None:
        """Verify strict validation raises when K-joint zeta is out of range."""
        geom = TubularJointGeometry(
            alpha= 8.0, beta=0.5, gamma=20.0, tau=0.6,
            theta_deg=45.0, joint_type=JointType.K,
            zeta=1.5  # above 1.0
        )
        with pytest.raises(ParameterRangeError, match="zeta"):
            validate_dnv_bounds(geom, strict=True)
 
    def test_boundary_values_are_valid(self) -> None:
        """Invariant: boundary values (exactly at limit) are valid per DNV."""
        geom = TubularJointGeometry(
            alpha= ALPHA_RANGE[0], beta=BETA_RANGE[0],    # exactly 0.2
            gamma=GAMMA_RANGE[1],  # exactly 30.0
            tau=TAU_RANGE[0],      # exactly 0.3
            theta_deg=THETA_DEG_RANGE[1],  # exactly 90°
            joint_type=JointType.T_Y
        )
        assert validate_dnv_bounds(geom, strict=False) == []
 
    def test_multiple_violations_collected_non_strict(self) -> None:
        """Non-strict mode must collect ALL violations, not stop at first."""
        geom = TubularJointGeometry(
            alpha= 7, 
            beta=0.1,   # bad
            gamma=35.0, # bad
            tau=0.6, theta_deg=60.0, joint_type=JointType.T_Y
        )
        violations = validate_dnv_bounds(geom, strict=False)
        assert len(violations) >= 3

class TestPhysicsInvariants:
    """Protect physical constraints that the DNV equations assume.
    
    These tests catch bugs where geometrically valid inputs produce
    physically nonsensical outputs.
    """

    def test_theta_rad_is_positive(self, valid_ty_geom: TubularJointGeometry) -> None:
        """Verify valid brace angles convert to positive radians."""
        assert valid_ty_geom.theta_rad > 0.0

    def test_theta_rad_does_not_exceed_pi_half_for_valid_input(
        self,
    ) -> None:
        """Verify a 90-degree brace angle converts to pi/2 radians."""
        geom_90 = TubularJointGeometry(
            alpha=12.0, beta=0.5, gamma=20.0, tau=0.6,
            theta_deg=90.0, joint_type=JointType.T_Y
        )
        assert math.isclose(geom_90.theta_rad, math.pi / 2)

    def test_beta_physically_meaningful(self) -> None:
        """Verify non-strict validation rejects physically invalid beta values."""
        geom = TubularJointGeometry(
            alpha=12.0, beta=1.2, gamma=20.0, tau=0.6,
            theta_deg=60.0, joint_type=JointType.T_Y
        )
        violations = validate_dnv_bounds(geom, strict=False)
        assert len(violations) > 0, "beta > 1.0 must be caught as a violation"
