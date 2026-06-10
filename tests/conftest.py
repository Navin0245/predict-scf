"""Shared pytest fixtures for all test levels.

Fixtures defined here are available to unit, integration, and e2e tests
without any import. pytest discovers this file automatically.

Fixture naming convention:
    valid_*_geom   -- TubularJointGeometry instances within DNV bounds
    invalid_*_geom -- instances outside DNV bounds (for error path tests)
    mock_model     -- lightweight model stub for API tests
"""
import pytest
