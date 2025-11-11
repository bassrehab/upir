"""
Unit tests for UPIR formal specification system.

Tests verify:
- FormalSpecification dataclass creation
- Validation of invariants, properties, and constraints
- Serialization/deserialization
- Hash computation and determinism
- Edge cases and error handling

Author: Subhadip Mitra
License: Apache 2.0
"""

import pytest
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty


class TestFormalSpecificationCreation:
    """Tests for creating FormalSpecification instances."""

    def test_create_empty_specification(self):
        """Test creating an empty specification."""
        spec = FormalSpecification()
        assert spec.invariants == []
        assert spec.properties == []
        assert spec.constraints == {}
        assert spec.assumptions == []

    def test_create_with_invariants(self):
        """Test creating specification with invariants."""
        inv1 = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        spec = FormalSpecification(invariants=[inv1])
        assert len(spec.invariants) == 1
        assert spec.invariants[0] == inv1

    def test_create_with_properties(self):
        """Test creating specification with properties."""
        prop1 = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="request_processed"
        )
        spec = FormalSpecification(properties=[prop1])
        assert len(spec.properties) == 1
        assert spec.properties[0] == prop1

    def test_create_with_constraints(self):
        """Test creating specification with constraints."""
        spec = FormalSpecification(
            constraints={
                "latency": {"max": 100},
                "cost_per_month": {"max": 10000}
            }
        )
        assert spec.constraints["latency"]["max"] == 100
        assert spec.constraints["cost_per_month"]["max"] == 10000

    def test_create_with_assumptions(self):
        """Test creating specification with assumptions."""
        spec = FormalSpecification(
            assumptions=["network_reliable", "nodes_fail_independently"]
        )
        assert len(spec.assumptions) == 2
        assert "network_reliable" in spec.assumptions

    def test_create_complete_specification(self):
        """Test creating a complete specification with all fields."""
        inv1 = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        prop1 = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="backup_complete",
            time_bound=3600.0
        )
        spec = FormalSpecification(
            invariants=[inv1],
            properties=[prop1],
            constraints={"latency": {"max": 100}},
            assumptions=["network_reliable"]
        )
        assert len(spec.invariants) == 1
        assert len(spec.properties) == 1
        assert len(spec.constraints) == 1
        assert len(spec.assumptions) == 1


class TestFormalSpecificationValidation:
    """Tests for specification validation."""

    def test_validate_empty_specification(self):
        """Test that empty specification is valid."""
        spec = FormalSpecification()
        assert spec.validate() is True

    def test_validate_simple_specification(self):
        """Test validation of simple specification."""
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="data_consistent"
                )
            ],
            constraints={"latency": {"max": 100}}
        )
        assert spec.validate() is True

    def test_validate_duplicate_invariants(self):
        """Test that duplicate invariants are rejected."""
        inv1 = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        inv2 = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        spec = FormalSpecification(invariants=[inv1, inv2])
        with pytest.raises(ValueError, match="Duplicate invariant"):
            spec.validate()

    def test_validate_duplicate_properties(self):
        """Test that duplicate properties are rejected."""
        prop1 = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="request_processed"
        )
        prop2 = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="request_processed"
        )
        spec = FormalSpecification(properties=[prop1, prop2])
        with pytest.raises(ValueError, match="Duplicate property"):
            spec.validate()

    def test_validate_same_predicate_different_operators_allowed(self):
        """Test that same predicate with different operators is allowed."""
        inv1 = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        inv2 = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="test"
        )
        spec = FormalSpecification(invariants=[inv1, inv2])
        assert spec.validate() is True

    def test_validate_constraint_with_max(self):
        """Test validation of constraint with max field."""
        spec = FormalSpecification(
            constraints={"latency": {"max": 100}}
        )
        assert spec.validate() is True

    def test_validate_constraint_with_min(self):
        """Test validation of constraint with min field."""
        spec = FormalSpecification(
            constraints={"throughput": {"min": 1000}}
        )
        assert spec.validate() is True

    def test_validate_constraint_with_equals(self):
        """Test validation of constraint with equals field."""
        spec = FormalSpecification(
            constraints={"replicas": {"equals": 3}}
        )
        assert spec.validate() is True

    def test_validate_constraint_with_min_and_max(self):
        """Test validation of constraint with both min and max."""
        spec = FormalSpecification(
            constraints={"latency": {"min": 10, "max": 100}}
        )
        assert spec.validate() is True

    def test_validate_constraint_min_greater_than_max(self):
        """Test that min > max is rejected."""
        spec = FormalSpecification(
            constraints={"latency": {"min": 100, "max": 10}}
        )
        with pytest.raises(ValueError, match="min .* > max"):
            spec.validate()

    def test_validate_constraint_without_valid_fields(self):
        """Test that constraint without min/max/equals is rejected."""
        spec = FormalSpecification(
            constraints={"latency": {"invalid_field": 100}}
        )
        with pytest.raises(ValueError, match="must have at least one of"):
            spec.validate()

    def test_validate_constraint_with_invalid_fields(self):
        """Test that constraint with extra invalid fields is rejected."""
        spec = FormalSpecification(
            constraints={"latency": {"max": 100, "invalid": 50}}
        )
        with pytest.raises(ValueError, match="has invalid fields"):
            spec.validate()

    def test_validate_constraint_equals_with_other_fields(self):
        """Test that 'equals' cannot be combined with min/max."""
        spec = FormalSpecification(
            constraints={"replicas": {"equals": 3, "min": 1}}
        )
        with pytest.raises(ValueError, match="'equals' cannot be combined"):
            spec.validate()

    def test_validate_constraint_not_dict(self):
        """Test that constraint must be a dictionary."""
        spec = FormalSpecification(
            constraints={"latency": 100}  # Should be {"max": 100}
        )
        with pytest.raises(ValueError, match="must be a dictionary"):
            spec.validate()


class TestFormalSpecificationSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict_empty_specification(self):
        """Test serialization of empty specification."""
        spec = FormalSpecification()
        data = spec.to_dict()
        assert data["invariants"] == []
        assert data["properties"] == []
        assert data["constraints"] == {}
        assert data["assumptions"] == []

    def test_to_dict_with_invariants(self):
        """Test serialization with invariants."""
        inv = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        spec = FormalSpecification(invariants=[inv])
        data = spec.to_dict()
        assert len(data["invariants"]) == 1
        assert data["invariants"][0]["operator"] == "ALWAYS"
        assert data["invariants"][0]["predicate"] == "data_consistent"

    def test_to_dict_with_constraints(self):
        """Test serialization with constraints."""
        spec = FormalSpecification(
            constraints={
                "latency": {"max": 100},
                "cost": {"max": 10000}
            }
        )
        data = spec.to_dict()
        assert data["constraints"]["latency"]["max"] == 100
        assert data["constraints"]["cost"]["max"] == 10000

    def test_to_dict_sorts_constraints(self):
        """Test that to_dict sorts constraints for determinism."""
        spec = FormalSpecification(
            constraints={
                "z_constraint": {"max": 3},
                "a_constraint": {"max": 1},
                "m_constraint": {"max": 2}
            }
        )
        data = spec.to_dict()
        keys = list(data["constraints"].keys())
        # Should be sorted alphabetically
        assert keys == ["a_constraint", "m_constraint", "z_constraint"]

    def test_to_dict_sorts_assumptions(self):
        """Test that to_dict sorts assumptions for determinism."""
        spec = FormalSpecification(
            assumptions=["zebra", "apple", "banana"]
        )
        data = spec.to_dict()
        assert data["assumptions"] == ["apple", "banana", "zebra"]

    def test_from_dict_empty_specification(self):
        """Test deserialization of empty specification."""
        data = {
            "invariants": [],
            "properties": [],
            "constraints": {},
            "assumptions": []
        }
        spec = FormalSpecification.from_dict(data)
        assert spec.invariants == []
        assert spec.properties == []
        assert spec.constraints == {}
        assert spec.assumptions == []

    def test_from_dict_with_invariants(self):
        """Test deserialization with invariants."""
        data = {
            "invariants": [
                {
                    "operator": "ALWAYS",
                    "predicate": "data_consistent",
                    "time_bound": None,
                    "parameters": {}
                }
            ],
            "properties": [],
            "constraints": {},
            "assumptions": []
        }
        spec = FormalSpecification.from_dict(data)
        assert len(spec.invariants) == 1
        assert spec.invariants[0].operator == TemporalOperator.ALWAYS

    def test_from_dict_with_constraints(self):
        """Test deserialization with constraints."""
        data = {
            "invariants": [],
            "properties": [],
            "constraints": {"latency": {"max": 100}},
            "assumptions": []
        }
        spec = FormalSpecification.from_dict(data)
        assert spec.constraints["latency"]["max"] == 100

    def test_from_dict_missing_optional_fields(self):
        """Test deserialization with missing optional fields."""
        data = {}
        spec = FormalSpecification.from_dict(data)
        assert spec.invariants == []
        assert spec.properties == []
        assert spec.constraints == {}
        assert spec.assumptions == []

    def test_roundtrip_serialization(self):
        """Test that serialize->deserialize preserves specification."""
        inv = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="backup_complete",
            time_bound=3600.0
        )
        original = FormalSpecification(
            invariants=[inv],
            properties=[prop],
            constraints={"latency": {"max": 100}},
            assumptions=["network_reliable"]
        )
        data = original.to_dict()
        restored = FormalSpecification.from_dict(data)

        assert len(restored.invariants) == len(original.invariants)
        assert len(restored.properties) == len(original.properties)
        assert restored.constraints == original.constraints
        assert set(restored.assumptions) == set(original.assumptions)


class TestFormalSpecificationHash:
    """Tests for hash computation."""

    def test_hash_empty_specification(self):
        """Test hash of empty specification."""
        spec = FormalSpecification()
        hash_val = spec.hash()
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA-256 produces 64 hex characters

    def test_hash_deterministic(self):
        """Test that hash is deterministic (same spec produces same hash)."""
        spec1 = FormalSpecification(
            constraints={"latency": {"max": 100}}
        )
        spec2 = FormalSpecification(
            constraints={"latency": {"max": 100}}
        )
        assert spec1.hash() == spec2.hash()

    def test_hash_different_for_different_specs(self):
        """Test that different specifications produce different hashes."""
        spec1 = FormalSpecification(
            constraints={"latency": {"max": 100}}
        )
        spec2 = FormalSpecification(
            constraints={"latency": {"max": 200}}
        )
        assert spec1.hash() != spec2.hash()

    def test_hash_independent_of_constraint_order(self):
        """Test that hash is independent of constraint insertion order."""
        spec1 = FormalSpecification(
            constraints={
                "latency": {"max": 100},
                "cost": {"max": 10000}
            }
        )
        spec2 = FormalSpecification(
            constraints={
                "cost": {"max": 10000},
                "latency": {"max": 100}
            }
        )
        # Hashes should be equal because to_dict sorts constraints
        assert spec1.hash() == spec2.hash()

    def test_hash_independent_of_assumption_order(self):
        """Test that hash is independent of assumption order."""
        spec1 = FormalSpecification(
            assumptions=["network_reliable", "nodes_independent"]
        )
        spec2 = FormalSpecification(
            assumptions=["nodes_independent", "network_reliable"]
        )
        # Hashes should be equal because to_dict sorts assumptions
        assert spec1.hash() == spec2.hash()

    def test_hash_with_complex_specification(self):
        """Test hash computation with complex specification."""
        inv = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="backup_complete",
            time_bound=3600.0
        )
        spec = FormalSpecification(
            invariants=[inv],
            properties=[prop],
            constraints={
                "latency": {"max": 100},
                "throughput": {"min": 1000}
            },
            assumptions=["network_reliable"]
        )
        hash_val = spec.hash()
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64


class TestFormalSpecificationStringRepresentation:
    """Tests for string representations."""

    def test_str_empty_specification(self):
        """Test __str__ for empty specification."""
        spec = FormalSpecification()
        assert str(spec) == "FormalSpecification(empty)"

    def test_str_with_invariants(self):
        """Test __str__ with invariants."""
        inv = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        spec = FormalSpecification(invariants=[inv])
        assert "1 invariant(s)" in str(spec)

    def test_str_with_multiple_fields(self):
        """Test __str__ with multiple fields."""
        inv = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        spec = FormalSpecification(
            invariants=[inv],
            constraints={"latency": {"max": 100}},
            assumptions=["network_reliable"]
        )
        s = str(spec)
        assert "1 invariant(s)" in s
        assert "1 constraint(s)" in s
        assert "1 assumption(s)" in s

    def test_repr(self):
        """Test __repr__ provides count information."""
        inv = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        spec = FormalSpecification(
            invariants=[inv],
            properties=[],
            constraints={"latency": {"max": 100}},
            assumptions=["network_reliable"]
        )
        repr_str = repr(spec)
        assert "FormalSpecification" in repr_str
        assert "invariants=1" in repr_str
        assert "properties=0" in repr_str
        assert "constraints=1" in repr_str
        assert "assumptions=1" in repr_str


class TestFormalSpecificationEdgeCases:
    """Tests for edge cases and corner cases."""

    def test_multiple_constraints_same_resource(self):
        """Test that we can have min and max for same resource."""
        spec = FormalSpecification(
            constraints={"latency": {"min": 10, "max": 100}}
        )
        assert spec.validate() is True

    def test_constraint_with_zero_values(self):
        """Test constraints with zero values."""
        spec = FormalSpecification(
            constraints={
                "min_latency": {"min": 0},
                "max_latency": {"max": 0}
            }
        )
        assert spec.validate() is True

    def test_constraint_with_negative_values(self):
        """Test constraints with negative values (e.g., for cost savings)."""
        spec = FormalSpecification(
            constraints={"cost_delta": {"min": -1000, "max": 0}}
        )
        assert spec.validate() is True

    def test_many_invariants(self):
        """Test specification with many invariants."""
        invariants = [
            TemporalProperty(
                operator=TemporalOperator.ALWAYS,
                predicate=f"invariant_{i}"
            )
            for i in range(100)
        ]
        spec = FormalSpecification(invariants=invariants)
        assert spec.validate() is True
        assert len(spec.invariants) == 100

    def test_complex_constraint_values(self):
        """Test constraints with complex values (floats, scientific notation)."""
        spec = FormalSpecification(
            constraints={
                "latency": {"max": 1e-3},  # 1 millisecond
                "throughput": {"min": 1e6}  # 1 million ops/sec
            }
        )
        assert spec.validate() is True

    def test_assumptions_with_special_characters(self):
        """Test assumptions with underscores and hyphens."""
        spec = FormalSpecification(
            assumptions=[
                "network_reliable",
                "nodes-fail-independently",
                "eventual_consistency_ok"
            ]
        )
        assert len(spec.assumptions) == 3
