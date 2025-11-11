"""
Unit tests for UPIR evidence tracking and Bayesian confidence updates.

Tests verify:
- Evidence creation and validation
- Bayesian confidence updates
- ReasoningNode creation and validation
- Geometric mean confidence computation
- Serialization/deserialization
- Edge cases and error handling

Author: Subhadip Mitra
License: Apache 2.0
"""

import math
import uuid
from datetime import datetime, timedelta

import pytest
from upir.core.evidence import Evidence, ReasoningNode


class TestEvidenceCreation:
    """Tests for creating Evidence instances."""

    def test_create_evidence_minimal(self):
        """Test creating evidence with minimal required fields."""
        evidence = Evidence(
            source="test_source",
            type="benchmark",
            data={"metric": 100},
            confidence=0.8
        )
        assert evidence.source == "test_source"
        assert evidence.type == "benchmark"
        assert evidence.data["metric"] == 100
        assert evidence.confidence == 0.8
        assert isinstance(evidence.timestamp, datetime)

    def test_create_evidence_all_types(self):
        """Test creating evidence with all valid types."""
        for evidence_type in ["benchmark", "test", "production", "formal_proof"]:
            evidence = Evidence(
                source="test",
                type=evidence_type,
                data={},
                confidence=0.5
            )
            assert evidence.type == evidence_type

    def test_create_evidence_with_timestamp(self):
        """Test creating evidence with explicit timestamp."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        evidence = Evidence(
            source="test",
            type="benchmark",
            data={},
            confidence=0.5,
            timestamp=ts
        )
        assert evidence.timestamp == ts

    def test_create_evidence_with_complex_data(self):
        """Test creating evidence with complex nested data."""
        data = {
            "metrics": {
                "latency_p50": 10,
                "latency_p99": 50,
                "throughput": 10000
            },
            "configuration": {
                "workers": 8,
                "cache_size": "1GB"
            }
        }
        evidence = Evidence(
            source="load_test",
            type="benchmark",
            data=data,
            confidence=0.9
        )
        assert evidence.data["metrics"]["latency_p99"] == 50

    def test_invalid_evidence_type(self):
        """Test that invalid evidence type is rejected."""
        with pytest.raises(ValueError, match="Invalid evidence type"):
            Evidence(
                source="test",
                type="invalid_type",
                data={},
                confidence=0.5
            )

    def test_confidence_out_of_range_low(self):
        """Test that confidence below 0 is rejected."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            Evidence(
                source="test",
                type="benchmark",
                data={},
                confidence=-0.1
            )

    def test_confidence_out_of_range_high(self):
        """Test that confidence above 1 is rejected."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            Evidence(
                source="test",
                type="benchmark",
                data={},
                confidence=1.1
            )

    def test_empty_source(self):
        """Test that empty source is rejected."""
        with pytest.raises(ValueError, match="Source cannot be empty"):
            Evidence(
                source="",
                type="benchmark",
                data={},
                confidence=0.5
            )

    def test_confidence_boundary_values(self):
        """Test confidence at boundary values 0 and 1."""
        evidence0 = Evidence(
            source="test",
            type="benchmark",
            data={},
            confidence=0.0
        )
        assert evidence0.confidence == 0.0

        evidence1 = Evidence(
            source="test",
            type="benchmark",
            data={},
            confidence=1.0
        )
        assert evidence1.confidence == 1.0


class TestEvidenceBayesianUpdate:
    """Tests for Bayesian confidence updates."""

    def test_positive_observation_increases_confidence(self):
        """Test that positive observation increases confidence."""
        evidence = Evidence(
            source="test",
            type="benchmark",
            data={},
            confidence=0.5
        )
        original_confidence = evidence.confidence
        evidence.update_confidence(new_observation=True, prior_weight=0.1)
        assert evidence.confidence > original_confidence

    def test_negative_observation_decreases_confidence(self):
        """Test that negative observation decreases confidence."""
        evidence = Evidence(
            source="test",
            type="benchmark",
            data={},
            confidence=0.5
        )
        original_confidence = evidence.confidence
        evidence.update_confidence(new_observation=False, prior_weight=0.1)
        assert evidence.confidence < original_confidence

    def test_positive_update_formula(self):
        """Test positive update formula: c += w * (1 - c)."""
        evidence = Evidence(
            source="test",
            type="benchmark",
            data={},
            confidence=0.5
        )
        evidence.update_confidence(new_observation=True, prior_weight=0.1)
        # Expected: 0.5 + 0.1 * (1 - 0.5) = 0.5 + 0.05 = 0.55
        assert abs(evidence.confidence - 0.55) < 1e-10

    def test_negative_update_formula(self):
        """Test negative update formula: c *= (1 - w)."""
        evidence = Evidence(
            source="test",
            type="benchmark",
            data={},
            confidence=0.5
        )
        evidence.update_confidence(new_observation=False, prior_weight=0.1)
        # Expected: 0.5 * (1 - 0.1) = 0.5 * 0.9 = 0.45
        assert abs(evidence.confidence - 0.45) < 1e-10

    def test_multiple_positive_updates_approach_one(self):
        """Test that multiple positive updates approach 1 asymptotically."""
        evidence = Evidence(
            source="test",
            type="benchmark",
            data={},
            confidence=0.5
        )
        for _ in range(100):
            evidence.update_confidence(new_observation=True, prior_weight=0.1)

        # Should approach 1 but never exceed it
        assert 0.99 < evidence.confidence <= 1.0

    def test_multiple_negative_updates_approach_zero(self):
        """Test that multiple negative updates approach 0."""
        evidence = Evidence(
            source="test",
            type="benchmark",
            data={},
            confidence=0.5
        )
        for _ in range(100):
            evidence.update_confidence(new_observation=False, prior_weight=0.1)

        # Should approach 0 but never go below it
        assert 0.0 <= evidence.confidence < 0.01

    def test_update_stays_in_valid_range(self):
        """Test that updates always keep confidence in [0, 1]."""
        evidence = Evidence(
            source="test",
            type="benchmark",
            data={},
            confidence=0.001
        )
        # Many negative updates from very low confidence
        for _ in range(50):
            evidence.update_confidence(new_observation=False, prior_weight=0.5)
            assert 0.0 <= evidence.confidence <= 1.0

        # Reset and test positive updates from high confidence
        evidence.confidence = 0.999
        for _ in range(50):
            evidence.update_confidence(new_observation=True, prior_weight=0.5)
            assert 0.0 <= evidence.confidence <= 1.0

    def test_different_prior_weights(self):
        """Test that different prior weights have different impacts."""
        e1 = Evidence("test", "benchmark", {}, 0.5)
        e2 = Evidence("test", "benchmark", {}, 0.5)

        e1.update_confidence(new_observation=True, prior_weight=0.1)
        e2.update_confidence(new_observation=True, prior_weight=0.5)

        # Higher prior weight should have larger impact
        assert e2.confidence > e1.confidence

    def test_prior_weight_validation(self):
        """Test that invalid prior_weight is rejected."""
        evidence = Evidence("test", "benchmark", {}, 0.5)

        with pytest.raises(ValueError, match="prior_weight must be in"):
            evidence.update_confidence(new_observation=True, prior_weight=-0.1)

        with pytest.raises(ValueError, match="prior_weight must be in"):
            evidence.update_confidence(new_observation=True, prior_weight=1.1)

    def test_zero_prior_weight_no_change(self):
        """Test that prior_weight=0 results in no change."""
        evidence = Evidence("test", "benchmark", {}, 0.5)
        evidence.update_confidence(new_observation=True, prior_weight=0.0)
        assert evidence.confidence == 0.5

    def test_update_at_boundary_zero(self):
        """Test update behavior when confidence is 0."""
        evidence = Evidence("test", "benchmark", {}, 0.0)

        # Negative update should keep it at 0
        evidence.update_confidence(new_observation=False, prior_weight=0.1)
        assert evidence.confidence == 0.0

        # Positive update should increase from 0
        evidence.update_confidence(new_observation=True, prior_weight=0.1)
        assert evidence.confidence > 0.0

    def test_update_at_boundary_one(self):
        """Test update behavior when confidence is 1."""
        evidence = Evidence("test", "benchmark", {}, 1.0)

        # Positive update should keep it at 1
        evidence.update_confidence(new_observation=True, prior_weight=0.1)
        assert evidence.confidence == 1.0

        # Negative update should decrease from 1
        evidence.update_confidence(new_observation=False, prior_weight=0.1)
        assert evidence.confidence < 1.0


class TestEvidenceSerialization:
    """Tests for evidence serialization."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        evidence = Evidence(
            source="test_source",
            type="benchmark",
            data={"metric": 100},
            confidence=0.8,
            timestamp=ts
        )
        d = evidence.to_dict()

        assert d["source"] == "test_source"
        assert d["type"] == "benchmark"
        assert d["data"]["metric"] == 100
        assert d["confidence"] == 0.8
        assert d["timestamp"] == "2024-01-01T12:00:00"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "source": "test_source",
            "type": "benchmark",
            "data": {"metric": 100},
            "confidence": 0.8,
            "timestamp": "2024-01-01T12:00:00"
        }
        evidence = Evidence.from_dict(data)

        assert evidence.source == "test_source"
        assert evidence.type == "benchmark"
        assert evidence.data["metric"] == 100
        assert evidence.confidence == 0.8
        assert evidence.timestamp == datetime(2024, 1, 1, 12, 0, 0)

    def test_roundtrip_serialization(self):
        """Test that serialize->deserialize preserves evidence."""
        original = Evidence(
            source="test",
            type="production",
            data={"cpu": 0.8, "memory": 0.6},
            confidence=0.75,
            timestamp=datetime(2024, 1, 1, 12, 0, 0)
        )
        data = original.to_dict()
        restored = Evidence.from_dict(data)

        assert restored.source == original.source
        assert restored.type == original.type
        assert restored.data == original.data
        assert restored.confidence == original.confidence
        assert restored.timestamp == original.timestamp

    def test_data_is_copied_in_to_dict(self):
        """Test that to_dict copies data (doesn't share reference)."""
        original_data = {"metric": 100}
        evidence = Evidence("test", "benchmark", original_data, 0.5)
        d = evidence.to_dict()
        d["data"]["metric"] = 200

        # Original data should be unchanged
        assert original_data["metric"] == 100
        assert evidence.data["metric"] == 100


class TestReasoningNodeCreation:
    """Tests for creating ReasoningNode instances."""

    def test_create_reasoning_node_minimal(self):
        """Test creating reasoning node with minimal fields."""
        node = ReasoningNode(
            id="node-1",
            decision="Use PostgreSQL",
            rationale="Strong consistency needed"
        )
        assert node.id == "node-1"
        assert node.decision == "Use PostgreSQL"
        assert node.rationale == "Strong consistency needed"
        assert node.evidence_ids == []
        assert node.parent_ids == []
        assert node.alternatives == []
        assert node.confidence == 0.0

    def test_create_reasoning_node_complete(self):
        """Test creating reasoning node with all fields."""
        node = ReasoningNode(
            id="node-1",
            decision="Use Redis for caching",
            rationale="Low latency access needed",
            evidence_ids=["e1", "e2"],
            parent_ids=["node-0"],
            alternatives=[
                {"option": "Memcached", "rejected_because": "Fewer features"},
                {"option": "No cache", "rejected_because": "Too slow"}
            ],
            confidence=0.85
        )
        assert len(node.evidence_ids) == 2
        assert len(node.parent_ids) == 1
        assert len(node.alternatives) == 2
        assert node.confidence == 0.85

    def test_generate_id(self):
        """Test ID generation using UUID."""
        node_id = ReasoningNode.generate_id()
        assert isinstance(node_id, str)
        assert len(node_id) == 36  # UUID format
        # Verify it's a valid UUID
        uuid.UUID(node_id)

    def test_generate_unique_ids(self):
        """Test that generated IDs are unique."""
        id1 = ReasoningNode.generate_id()
        id2 = ReasoningNode.generate_id()
        assert id1 != id2

    def test_empty_id_rejected(self):
        """Test that empty ID is rejected."""
        with pytest.raises(ValueError, match="ID cannot be empty"):
            ReasoningNode(
                id="",
                decision="test",
                rationale="test"
            )

    def test_empty_decision_rejected(self):
        """Test that empty decision is rejected."""
        with pytest.raises(ValueError, match="Decision cannot be empty"):
            ReasoningNode(
                id="node-1",
                decision="",
                rationale="test"
            )

    def test_confidence_out_of_range(self):
        """Test that invalid confidence is rejected."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            ReasoningNode(
                id="node-1",
                decision="test",
                rationale="test",
                confidence=1.5
            )


class TestReasoningNodeConfidenceComputation:
    """Tests for geometric mean confidence computation."""

    def test_compute_confidence_single_evidence(self):
        """Test confidence computation with single evidence."""
        evidence_map = {
            "e1": Evidence("src", "test", {}, 0.8, datetime.utcnow())
        }
        node = ReasoningNode(
            id="node-1",
            decision="test",
            rationale="test",
            evidence_ids=["e1"]
        )
        conf = node.compute_confidence(evidence_map)
        assert abs(conf - 0.8) < 1e-10

    def test_compute_confidence_multiple_evidence(self):
        """Test confidence computation with multiple evidence."""
        evidence_map = {
            "e1": Evidence("src1", "test", {}, 0.8, datetime.utcnow()),
            "e2": Evidence("src2", "test", {}, 0.9, datetime.utcnow())
        }
        node = ReasoningNode(
            id="node-1",
            decision="test",
            rationale="test",
            evidence_ids=["e1", "e2"]
        )
        conf = node.compute_confidence(evidence_map)
        # Geometric mean: sqrt(0.8 * 0.9) ≈ 0.8485
        expected = math.sqrt(0.8 * 0.9)
        assert abs(conf - expected) < 1e-10

    def test_geometric_mean_formula(self):
        """Test geometric mean formula: exp(mean(log(c_i)))."""
        confidences = [0.7, 0.8, 0.9]
        evidence_map = {
            f"e{i}": Evidence(f"src{i}", "test", {}, c, datetime.utcnow())
            for i, c in enumerate(confidences)
        }
        node = ReasoningNode(
            id="node-1",
            decision="test",
            rationale="test",
            evidence_ids=list(evidence_map.keys())
        )
        conf = node.compute_confidence(evidence_map)

        # Manual computation: exp(mean(log(c_i)))
        log_sum = sum(math.log(c) for c in confidences)
        log_mean = log_sum / len(confidences)
        expected = math.exp(log_mean)

        assert abs(conf - expected) < 1e-10

    def test_geometric_mean_more_conservative(self):
        """Test that geometric mean is more conservative than arithmetic."""
        # Confidences: one low, two high
        evidence_map = {
            "e1": Evidence("src1", "test", {}, 0.3, datetime.utcnow()),
            "e2": Evidence("src2", "test", {}, 0.9, datetime.utcnow()),
            "e3": Evidence("src3", "test", {}, 0.9, datetime.utcnow())
        }
        node = ReasoningNode(
            id="node-1",
            decision="test",
            rationale="test",
            evidence_ids=["e1", "e2", "e3"]
        )
        geometric = node.compute_confidence(evidence_map)
        arithmetic = (0.3 + 0.9 + 0.9) / 3  # = 0.7

        # Geometric mean should be lower (more conservative)
        assert geometric < arithmetic

    def test_no_evidence_returns_zero(self):
        """Test that no evidence returns 0 confidence."""
        node = ReasoningNode(
            id="node-1",
            decision="test",
            rationale="test",
            evidence_ids=[]
        )
        conf = node.compute_confidence({})
        assert conf == 0.0

    def test_missing_evidence_ignored(self):
        """Test that missing evidence IDs are ignored."""
        evidence_map = {
            "e1": Evidence("src1", "test", {}, 0.8, datetime.utcnow())
        }
        node = ReasoningNode(
            id="node-1",
            decision="test",
            rationale="test",
            evidence_ids=["e1", "e2", "e3"]  # e2, e3 don't exist
        )
        conf = node.compute_confidence(evidence_map)
        # Should only use e1
        assert abs(conf - 0.8) < 1e-10

    def test_all_evidence_missing_returns_zero(self):
        """Test that all missing evidence returns 0."""
        node = ReasoningNode(
            id="node-1",
            decision="test",
            rationale="test",
            evidence_ids=["e1", "e2"]
        )
        conf = node.compute_confidence({})  # Empty evidence map
        assert conf == 0.0

    def test_zero_confidence_evidence_returns_zero(self):
        """Test that any zero confidence evidence makes result zero."""
        evidence_map = {
            "e1": Evidence("src1", "test", {}, 0.0, datetime.utcnow()),
            "e2": Evidence("src2", "test", {}, 0.9, datetime.utcnow())
        }
        node = ReasoningNode(
            id="node-1",
            decision="test",
            rationale="test",
            evidence_ids=["e1", "e2"]
        )
        conf = node.compute_confidence(evidence_map)
        assert conf == 0.0


class TestReasoningNodeSerialization:
    """Tests for reasoning node serialization."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        node = ReasoningNode(
            id="node-1",
            decision="Use Redis",
            rationale="Low latency",
            evidence_ids=["e1", "e2"],
            parent_ids=["node-0"],
            alternatives=[{"option": "Memcached"}],
            confidence=0.85
        )
        d = node.to_dict()

        assert d["id"] == "node-1"
        assert d["decision"] == "Use Redis"
        assert d["rationale"] == "Low latency"
        assert d["evidence_ids"] == ["e1", "e2"]
        assert d["parent_ids"] == ["node-0"]
        assert len(d["alternatives"]) == 1
        assert d["confidence"] == 0.85

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "node-1",
            "decision": "Use Redis",
            "rationale": "Low latency",
            "evidence_ids": ["e1"],
            "parent_ids": ["node-0"],
            "alternatives": [{"option": "Memcached"}],
            "confidence": 0.85
        }
        node = ReasoningNode.from_dict(data)

        assert node.id == "node-1"
        assert node.decision == "Use Redis"
        assert node.evidence_ids == ["e1"]
        assert node.confidence == 0.85

    def test_from_dict_missing_optional_fields(self):
        """Test deserialization with missing optional fields."""
        data = {
            "id": "node-1",
            "decision": "Use Redis",
            "rationale": "Low latency"
        }
        node = ReasoningNode.from_dict(data)

        assert node.evidence_ids == []
        assert node.parent_ids == []
        assert node.alternatives == []
        assert node.confidence == 0.0

    def test_roundtrip_serialization(self):
        """Test that serialize->deserialize preserves node."""
        original = ReasoningNode(
            id="node-1",
            decision="Use PostgreSQL",
            rationale="ACID compliance",
            evidence_ids=["e1", "e2"],
            parent_ids=["node-0"],
            alternatives=[
                {"option": "MySQL", "reason": "Less features"}
            ],
            confidence=0.9
        )
        data = original.to_dict()
        restored = ReasoningNode.from_dict(data)

        assert restored.id == original.id
        assert restored.decision == original.decision
        assert restored.rationale == original.rationale
        assert restored.evidence_ids == original.evidence_ids
        assert restored.parent_ids == original.parent_ids
        assert restored.confidence == original.confidence

    def test_lists_are_copied(self):
        """Test that to_dict copies lists (doesn't share references)."""
        node = ReasoningNode(
            id="node-1",
            decision="test",
            rationale="test",
            evidence_ids=["e1"],
            parent_ids=["p1"],
            alternatives=[{"option": "alt"}]
        )
        d = node.to_dict()

        # Modify returned lists
        d["evidence_ids"].append("e2")
        d["parent_ids"].append("p2")
        d["alternatives"].append({"option": "alt2"})

        # Original should be unchanged
        assert node.evidence_ids == ["e1"]
        assert node.parent_ids == ["p1"]
        assert len(node.alternatives) == 1


class TestStringRepresentations:
    """Tests for string representations."""

    def test_evidence_str(self):
        """Test Evidence __str__ method."""
        evidence = Evidence("test_src", "benchmark", {}, 0.85, datetime.utcnow())
        s = str(evidence)
        assert "benchmark" in s
        assert "test_src" in s
        assert "0.85" in s

    def test_reasoning_node_str(self):
        """Test ReasoningNode __str__ method."""
        node = ReasoningNode(
            id="node-1",
            decision="Use Redis",
            rationale="test",
            evidence_ids=["e1", "e2"],
            confidence=0.8
        )
        s = str(node)
        assert "Use Redis" in s
        assert "0.80" in s
        assert "2" in s  # evidence count

    def test_reasoning_node_repr(self):
        """Test ReasoningNode __repr__ method."""
        node = ReasoningNode(
            id="node-1",
            decision="Use Redis",
            rationale="test",
            evidence_ids=["e1"],
            parent_ids=["p1"]
        )
        r = repr(node)
        assert "ReasoningNode" in r
        assert "node-1" in r
        assert "Use Redis" in r
        assert "evidence_count=1" in r
        assert "parent_count=1" in r
