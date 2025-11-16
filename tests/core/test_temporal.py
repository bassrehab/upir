"""
Unit tests for UPIR temporal property system.

Tests verify:
- TemporalOperator enum values
- TemporalProperty dataclass validation
- SMT-LIB encoding correctness
- Serialization/deserialization
- Edge cases and error handling

Author: Subhadip Mitra
License: Apache 2.0
"""

import pytest

from upir.core.temporal import TemporalOperator, TemporalProperty


class TestTemporalOperator:
    """Tests for TemporalOperator enum."""

    def test_operator_values(self):
        """Test that all expected operators exist with correct values."""
        assert TemporalOperator.ALWAYS.value == "ALWAYS"
        assert TemporalOperator.EVENTUALLY.value == "EVENTUALLY"
        assert TemporalOperator.WITHIN.value == "WITHIN"
        assert TemporalOperator.UNTIL.value == "UNTIL"

    def test_operator_from_string(self):
        """Test creating operators from string values."""
        assert TemporalOperator("ALWAYS") == TemporalOperator.ALWAYS
        assert TemporalOperator("EVENTUALLY") == TemporalOperator.EVENTUALLY
        assert TemporalOperator("WITHIN") == TemporalOperator.WITHIN
        assert TemporalOperator("UNTIL") == TemporalOperator.UNTIL


class TestTemporalProperty:
    """Tests for TemporalProperty dataclass."""

    def test_create_always_property(self):
        """Test creating ALWAYS temporal property."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        assert prop.operator == TemporalOperator.ALWAYS
        assert prop.predicate == "data_consistent"
        assert prop.time_bound is None
        assert prop.parameters == {}

    def test_create_eventually_property(self):
        """Test creating EVENTUALLY temporal property."""
        prop = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="request_processed"
        )
        assert prop.operator == TemporalOperator.EVENTUALLY
        assert prop.predicate == "request_processed"

    def test_create_within_property(self):
        """Test creating WITHIN temporal property with time bound."""
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="backup_complete",
            time_bound=3600.0
        )
        assert prop.operator == TemporalOperator.WITHIN
        assert prop.predicate == "backup_complete"
        assert prop.time_bound == 3600.0

    def test_create_until_property(self):
        """Test creating UNTIL temporal property."""
        prop = TemporalProperty(
            operator=TemporalOperator.UNTIL,
            predicate="processing"
        )
        assert prop.operator == TemporalOperator.UNTIL
        assert prop.predicate == "processing"

    def test_property_with_parameters(self):
        """Test property with additional parameters."""
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="response_received",
            time_bound=0.1,
            parameters={"max_latency_ms": 100, "percentile": 99}
        )
        assert prop.parameters["max_latency_ms"] == 100
        assert prop.parameters["percentile"] == 99

    def test_within_requires_time_bound(self):
        """Test that WITHIN operator requires a time_bound."""
        with pytest.raises(ValueError, match="WITHIN operator requires a time_bound"):
            TemporalProperty(
                operator=TemporalOperator.WITHIN,
                predicate="test"
            )

    def test_time_bound_must_be_positive(self):
        """Test that time_bound must be positive."""
        with pytest.raises(ValueError, match="time_bound must be positive"):
            TemporalProperty(
                operator=TemporalOperator.WITHIN,
                predicate="test",
                time_bound=0
            )

        with pytest.raises(ValueError, match="time_bound must be positive"):
            TemporalProperty(
                operator=TemporalOperator.WITHIN,
                predicate="test",
                time_bound=-1.0
            )

    def test_predicate_cannot_be_empty(self):
        """Test that predicate cannot be empty string."""
        with pytest.raises(ValueError, match="predicate cannot be empty"):
            TemporalProperty(
                operator=TemporalOperator.ALWAYS,
                predicate=""
            )


class TestTemporalPropertySMT:
    """Tests for SMT-LIB encoding of temporal properties."""

    def test_always_to_smt(self):
        """Test ALWAYS operator converts to universal quantification."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        smt = prop.to_smt()
        assert "forall" in smt
        assert "data_consistent" in smt
        assert smt == "(forall ((t Real)) (data_consistent t))"

    def test_eventually_to_smt(self):
        """Test EVENTUALLY operator converts to existential quantification."""
        prop = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="request_processed"
        )
        smt = prop.to_smt()
        assert "exists" in smt
        assert "request_processed" in smt
        assert smt == "(exists ((t Real)) (request_processed t))"

    def test_within_to_smt(self):
        """Test WITHIN operator converts to bounded existential."""
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="backup_complete",
            time_bound=3600.0
        )
        smt = prop.to_smt()
        assert "exists" in smt
        assert "backup_complete" in smt
        assert "<= t 3600.0" in smt or "<= t 3600" in smt
        assert "and" in smt

    def test_until_to_smt(self):
        """Test UNTIL operator converts to appropriate SMT formula."""
        prop = TemporalProperty(
            operator=TemporalOperator.UNTIL,
            predicate="processing"
        )
        smt = prop.to_smt()
        assert "exists" in smt
        assert "forall" in smt
        assert "processing" in smt
        # UNTIL encoding: ∃t. Q(t) ∧ ∀s. (s < t) → P(s)
        assert "processing_q" in smt
        assert "processing_p" in smt


class TestTemporalPropertySerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict_always(self):
        """Test serialization of ALWAYS property."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        data = prop.to_dict()
        assert data["operator"] == "ALWAYS"
        assert data["predicate"] == "data_consistent"
        assert data["time_bound"] is None
        assert data["parameters"] == {}

    def test_to_dict_within(self):
        """Test serialization of WITHIN property."""
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="backup_complete",
            time_bound=3600.0,
            parameters={"backup_type": "full"}
        )
        data = prop.to_dict()
        assert data["operator"] == "WITHIN"
        assert data["predicate"] == "backup_complete"
        assert data["time_bound"] == 3600.0
        assert data["parameters"]["backup_type"] == "full"

    def test_from_dict_always(self):
        """Test deserialization of ALWAYS property."""
        data = {
            "operator": "ALWAYS",
            "predicate": "data_consistent",
            "time_bound": None,
            "parameters": {}
        }
        prop = TemporalProperty.from_dict(data)
        assert prop.operator == TemporalOperator.ALWAYS
        assert prop.predicate == "data_consistent"
        assert prop.time_bound is None
        assert prop.parameters == {}

    def test_from_dict_within(self):
        """Test deserialization of WITHIN property."""
        data = {
            "operator": "WITHIN",
            "predicate": "backup_complete",
            "time_bound": 3600.0,
            "parameters": {"backup_type": "full"}
        }
        prop = TemporalProperty.from_dict(data)
        assert prop.operator == TemporalOperator.WITHIN
        assert prop.predicate == "backup_complete"
        assert prop.time_bound == 3600.0
        assert prop.parameters["backup_type"] == "full"

    def test_roundtrip_serialization(self):
        """Test that serialize->deserialize preserves property."""
        original = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="response_received",
            time_bound=0.1,
            parameters={"max_latency_ms": 100}
        )
        data = original.to_dict()
        restored = TemporalProperty.from_dict(data)

        assert restored.operator == original.operator
        assert restored.predicate == original.predicate
        assert restored.time_bound == original.time_bound
        assert restored.parameters == original.parameters

    def test_from_dict_missing_parameters(self):
        """Test deserialization with missing optional parameters field."""
        data = {
            "operator": "ALWAYS",
            "predicate": "data_consistent"
        }
        prop = TemporalProperty.from_dict(data)
        assert prop.parameters == {}


class TestTemporalPropertyStringRepresentation:
    """Tests for string representations."""

    def test_str_always(self):
        """Test __str__ for ALWAYS property."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        assert str(prop) == "ALWAYS(data_consistent)"

    def test_str_within(self):
        """Test __str__ for WITHIN property with time bound."""
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="backup_complete",
            time_bound=3600.0
        )
        assert str(prop) == "WITHIN[3600.0s](backup_complete)"

    def test_repr(self):
        """Test __repr__ provides complete information."""
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="test",
            time_bound=10.0,
            parameters={"key": "value"}
        )
        repr_str = repr(prop)
        assert "TemporalProperty" in repr_str
        assert "WITHIN" in repr_str
        assert "test" in repr_str
        assert "10.0" in repr_str


class TestTemporalPropertyEdgeCases:
    """Tests for edge cases and corner cases."""

    def test_very_large_time_bound(self):
        """Test property with very large time bound."""
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="test",
            time_bound=1e10  # ~317 years
        )
        assert prop.time_bound == 1e10

    def test_very_small_time_bound(self):
        """Test property with very small time bound."""
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="test",
            time_bound=1e-9  # 1 nanosecond
        )
        assert prop.time_bound == 1e-9

    def test_predicate_with_special_characters(self):
        """Test predicate with underscores and numbers."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="check_status_v2"
        )
        assert prop.predicate == "check_status_v2"

    def test_complex_parameters(self):
        """Test property with complex nested parameters."""
        prop = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="test",
            parameters={
                "thresholds": {"cpu": 0.8, "memory": 0.9},
                "regions": ["us-west", "us-east"],
                "enabled": True
            }
        )
        assert prop.parameters["thresholds"]["cpu"] == 0.8
        assert "us-west" in prop.parameters["regions"]
        assert prop.parameters["enabled"] is True

    def test_parameters_are_copied_in_to_dict(self):
        """Test that to_dict copies parameters (doesn't share reference)."""
        original_params = {"key": "value"}
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test",
            parameters=original_params
        )
        data = prop.to_dict()
        data["parameters"]["key"] = "modified"

        # Original parameters should be unchanged
        assert original_params["key"] == "value"
        assert prop.parameters["key"] == "value"
