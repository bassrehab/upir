"""
Unit tests for SMT-based verification using Z3.

Tests verify:
- VerificationStatus enum
- ProofCertificate creation and hashing
- VerificationResult creation and properties
- Serialization/deserialization
- Z3 availability checks

Author: Subhadip Mitra
License: Apache 2.0
"""

from datetime import datetime

import pytest

from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.verification.solver import (
    ProofCertificate,
    VerificationResult,
    VerificationStatus,
    get_z3_version,
    is_z3_available,
)


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert VerificationStatus.PROVED.value == "PROVED"
        assert VerificationStatus.DISPROVED.value == "DISPROVED"
        assert VerificationStatus.UNKNOWN.value == "UNKNOWN"
        assert VerificationStatus.TIMEOUT.value == "TIMEOUT"
        assert VerificationStatus.ERROR.value == "ERROR"

    def test_status_from_string(self):
        """Test creating status from string value."""
        assert VerificationStatus("PROVED") == VerificationStatus.PROVED
        assert VerificationStatus("DISPROVED") == VerificationStatus.DISPROVED
        assert VerificationStatus("UNKNOWN") == VerificationStatus.UNKNOWN
        assert VerificationStatus("TIMEOUT") == VerificationStatus.TIMEOUT
        assert VerificationStatus("ERROR") == VerificationStatus.ERROR

    def test_invalid_status(self):
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError):
            VerificationStatus("INVALID")


class TestProofCertificate:
    """Tests for ProofCertificate dataclass."""

    def test_create_minimal_certificate(self):
        """Test creating certificate with minimal fields."""
        cert = ProofCertificate(
            property_hash="abc123",
            architecture_hash="def456",
            status=VerificationStatus.PROVED
        )
        assert cert.property_hash == "abc123"
        assert cert.architecture_hash == "def456"
        assert cert.status == VerificationStatus.PROVED
        assert cert.proof_steps == []
        assert cert.assumptions == []
        assert isinstance(cert.timestamp, datetime)

    def test_create_complete_certificate(self):
        """Test creating certificate with all fields."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        cert = ProofCertificate(
            property_hash="abc123",
            architecture_hash="def456",
            status=VerificationStatus.PROVED,
            proof_steps=[
                {"step": 1, "action": "simplify"},
                {"step": 2, "action": "solve"}
            ],
            assumptions=["network_reliable", "no_failures"],
            timestamp=ts,
            solver_version="z3-4.12.2"
        )
        assert len(cert.proof_steps) == 2
        assert len(cert.assumptions) == 2
        assert cert.timestamp == ts
        assert cert.solver_version == "z3-4.12.2"

    def test_certificate_hash_generation(self):
        """Test certificate hash generation."""
        cert = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            solver_version="z3-4.12.2"
        )
        hash_val = cert.generate_hash()

        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA-256

    def test_certificate_hash_deterministic(self):
        """Test that hash generation is deterministic."""
        cert = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            solver_version="z3-4.12.2"
        )
        hash1 = cert.generate_hash()
        hash2 = cert.generate_hash()

        assert hash1 == hash2

    def test_certificate_hash_changes_with_content(self):
        """Test that hash changes when content changes."""
        cert1 = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            solver_version="z3-4.12.2"
        )
        cert2 = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.DISPROVED,  # Different status
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            solver_version="z3-4.12.2"
        )
        assert cert1.generate_hash() != cert2.generate_hash()

    def test_certificate_hash_independent_of_assumption_order(self):
        """Test that hash is independent of assumption order."""
        cert1 = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED,
            assumptions=["a", "b", "c"],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            solver_version="z3-4.12.2"
        )
        cert2 = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED,
            assumptions=["c", "b", "a"],  # Different order
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            solver_version="z3-4.12.2"
        )
        # Should be equal because assumptions are sorted in hash
        assert cert1.generate_hash() == cert2.generate_hash()

    def test_certificate_to_dict(self):
        """Test certificate serialization."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        cert = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED,
            proof_steps=[{"step": 1}],
            assumptions=["network_reliable"],
            timestamp=ts,
            solver_version="z3-4.12.2"
        )
        d = cert.to_dict()

        assert d["property_hash"] == "abc"
        assert d["architecture_hash"] == "def"
        assert d["status"] == "PROVED"
        assert len(d["proof_steps"]) == 1
        assert d["assumptions"] == ["network_reliable"]
        assert d["timestamp"] == "2024-01-01T12:00:00"
        assert d["solver_version"] == "z3-4.12.2"

    def test_certificate_from_dict(self):
        """Test certificate deserialization."""
        data = {
            "property_hash": "abc",
            "architecture_hash": "def",
            "status": "PROVED",
            "proof_steps": [{"step": 1}],
            "assumptions": ["network_reliable"],
            "timestamp": "2024-01-01T12:00:00",
            "solver_version": "z3-4.12.2"
        }
        cert = ProofCertificate.from_dict(data)

        assert cert.property_hash == "abc"
        assert cert.status == VerificationStatus.PROVED
        assert len(cert.proof_steps) == 1
        assert cert.timestamp == datetime(2024, 1, 1, 12, 0, 0)

    def test_certificate_roundtrip(self):
        """Test that serialize->deserialize preserves certificate."""
        original = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED,
            proof_steps=[{"step": 1, "action": "solve"}],
            assumptions=["a1"],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            solver_version="z3-4.12.2"
        )
        data = original.to_dict()
        restored = ProofCertificate.from_dict(data)

        assert restored.property_hash == original.property_hash
        assert restored.architecture_hash == original.architecture_hash
        assert restored.status == original.status
        assert restored.proof_steps == original.proof_steps
        assert restored.assumptions == original.assumptions
        assert restored.timestamp == original.timestamp
        assert restored.solver_version == original.solver_version

    def test_certificate_str(self):
        """Test __str__ representation."""
        cert = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED,
            solver_version="z3-4.12.2"
        )
        s = str(cert)
        assert "PROVED" in s
        assert "z3-4.12.2" in s


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_create_minimal_result(self):
        """Test creating result with minimal fields."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED
        )
        assert result.property == prop
        assert result.status == VerificationStatus.PROVED
        assert result.certificate is None
        assert result.counterexample is None
        assert result.execution_time == 0.0
        assert result.cached is False

    def test_create_complete_result(self):
        """Test creating result with all fields."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        cert = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED
        )
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED,
            certificate=cert,
            counterexample=None,
            execution_time=1.23,
            cached=True
        )
        assert result.certificate is not None
        assert result.execution_time == 1.23
        assert result.cached is True

    def test_result_with_counterexample(self):
        """Test result with counterexample (disproved)."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        counterexample = {
            "state": {"node1": "failed", "node2": "ok"},
            "time": 42
        }
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.DISPROVED,
            counterexample=counterexample,
            execution_time=0.5
        )
        assert result.status == VerificationStatus.DISPROVED
        assert result.counterexample is not None
        assert result.counterexample["time"] == 42

    def test_verified_property_true(self):
        """Test verified property returns True for PROVED."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED
        )
        assert result.verified is True

    def test_verified_property_false(self):
        """Test verified property returns False for non-PROVED."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )

        for status in [
            VerificationStatus.DISPROVED,
            VerificationStatus.UNKNOWN,
            VerificationStatus.TIMEOUT,
            VerificationStatus.ERROR
        ]:
            result = VerificationResult(property=prop, status=status)
            assert result.verified is False

    def test_result_to_dict(self):
        """Test result serialization."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED,
            execution_time=1.23,
            cached=True
        )
        d = result.to_dict()

        assert d["status"] == "PROVED"
        assert d["property"]["predicate"] == "test"
        assert d["execution_time"] == 1.23
        assert d["cached"] is True

    def test_result_from_dict(self):
        """Test result deserialization."""
        data = {
            "property": {
                "operator": "ALWAYS",
                "predicate": "test",
                "time_bound": None,
                "parameters": {}
            },
            "status": "PROVED",
            "certificate": None,
            "counterexample": None,
            "execution_time": 1.23,
            "cached": True
        }
        result = VerificationResult.from_dict(data)

        assert result.status == VerificationStatus.PROVED
        assert result.property.predicate == "test"
        assert result.execution_time == 1.23
        assert result.cached is True

    def test_result_roundtrip(self):
        """Test that serialize->deserialize preserves result."""
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="backup_complete",
            time_bound=3600.0
        )
        cert = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            solver_version="z3-4.12.2"
        )
        original = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED,
            certificate=cert,
            execution_time=2.5,
            cached=False
        )
        data = original.to_dict()
        restored = VerificationResult.from_dict(data)

        assert restored.status == original.status
        assert restored.property.predicate == original.property.predicate
        assert restored.execution_time == original.execution_time
        assert restored.cached == original.cached
        assert restored.certificate is not None

    def test_result_str(self):
        """Test __str__ representation."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED,
            execution_time=1.23,
            cached=True
        )
        s = str(result)
        assert "PROVED" in s
        assert "test" in s
        assert "cached=True" in s
        assert "1.23" in s

    def test_result_repr(self):
        """Test __repr__ representation."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED,
            cached=True
        )
        r = repr(result)
        assert "VerificationResult" in r
        assert "PROVED" in r
        assert "test" in r
        assert "verified=True" in r
        assert "cached=True" in r


class TestZ3Availability:
    """Tests for Z3 availability checks."""

    def test_is_z3_available(self):
        """Test Z3 availability check."""
        # Should return True since we installed z3-solver
        available = is_z3_available()
        assert isinstance(available, bool)
        # In our test environment, should be True
        assert available is True

    def test_get_z3_version(self):
        """Test getting Z3 version."""
        version = get_z3_version()
        assert isinstance(version, str)
        # Should have version info since Z3 is installed
        assert "4.12" in version or version == "unavailable"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_certificate_with_empty_assumptions(self):
        """Test certificate with no assumptions."""
        cert = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED,
            assumptions=[]
        )
        assert cert.assumptions == []
        hash_val = cert.generate_hash()
        assert len(hash_val) == 64

    def test_certificate_with_many_proof_steps(self):
        """Test certificate with many proof steps."""
        steps = [{"step": i, "action": f"step{i}"} for i in range(100)]
        cert = ProofCertificate(
            property_hash="abc",
            architecture_hash="def",
            status=VerificationStatus.PROVED,
            proof_steps=steps
        )
        assert len(cert.proof_steps) == 100

    def test_result_unknown_status(self):
        """Test result with UNKNOWN status."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.UNKNOWN,
            execution_time=10.0  # Took time but couldn't determine
        )
        assert result.verified is False
        assert result.status == VerificationStatus.UNKNOWN

    def test_result_timeout_status(self):
        """Test result with TIMEOUT status."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.TIMEOUT,
            execution_time=60.0
        )
        assert result.verified is False
        assert result.status == VerificationStatus.TIMEOUT

    def test_result_error_status(self):
        """Test result with ERROR status."""
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.ERROR
        )
        assert result.verified is False
        assert result.status == VerificationStatus.ERROR

    def test_certificate_from_dict_missing_optional_fields(self):
        """Test certificate deserialization with missing optional fields."""
        data = {
            "property_hash": "abc",
            "architecture_hash": "def",
            "status": "PROVED",
            "timestamp": "2024-01-01T12:00:00"
        }
        cert = ProofCertificate.from_dict(data)
        assert cert.proof_steps == []
        assert cert.assumptions == []
        assert cert.solver_version == "unknown"
