"""
Unit tests for Pattern dataclass.

Tests verify:
- Pattern creation and validation
- Instance management
- Similarity matching
- Serialization

Author: Subhadip Mitra
License: Apache 2.0
"""

import numpy as np
import pytest

from upir.patterns.pattern import Pattern


class TestPatternCreation:
    """Tests for Pattern creation."""

    def test_create_minimal_pattern(self):
        """Test creating pattern with minimal fields."""
        pattern = Pattern(
            id="p1",
            name="Test Pattern",
            description="A test pattern",
            template={"components": []}
        )
        assert pattern.id == "p1"
        assert pattern.name == "Test Pattern"
        assert len(pattern.instances) == 0
        assert pattern.success_rate == 0.0

    def test_create_complete_pattern(self):
        """Test creating pattern with all fields."""
        pattern = Pattern(
            id="streaming-etl",
            name="Streaming ETL",
            description="Event-driven pipeline",
            template={
                "components": [{"type": "pubsub"}],
                "parameters": {"window_size": 60}
            },
            instances=["upir-1", "upir-2"],
            success_rate=0.95,
            average_performance={"latency_p99": 100}
        )
        assert pattern.id == "streaming-etl"
        assert len(pattern.instances) == 2
        assert pattern.success_rate == 0.95
        assert pattern.average_performance["latency_p99"] == 100

    def test_empty_id_raises_error(self):
        """Test that empty ID raises ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            Pattern(id="", name="Test", description="Test", template={})

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Pattern(id="p1", name="", description="Test", template={})

    def test_invalid_success_rate_raises_error(self):
        """Test that invalid success rate raises ValueError."""
        with pytest.raises(ValueError, match="Success rate must be in"):
            Pattern(
                id="p1",
                name="Test",
                description="Test",
                template={},
                success_rate=1.5  # Invalid
            )


class TestInstanceManagement:
    """Tests for managing pattern instances."""

    def test_add_instance(self):
        """Test adding an instance."""
        pattern = Pattern(id="p1", name="Test", description="Test", template={})

        pattern.add_instance("upir-1")

        assert "upir-1" in pattern.instances
        assert len(pattern.instances) == 1

    def test_add_instance_with_performance(self):
        """Test adding instance with performance metrics."""
        pattern = Pattern(id="p1", name="Test", description="Test", template={})

        pattern.add_instance("upir-1", performance={"latency_p99": 100})

        assert "upir-1" in pattern.instances
        assert pattern.average_performance["latency_p99"] == 100

    def test_add_multiple_instances_updates_average(self):
        """Test that adding instances updates average performance."""
        pattern = Pattern(id="p1", name="Test", description="Test", template={})

        pattern.add_instance("upir-1", performance={"latency_p99": 100})
        pattern.add_instance("upir-2", performance={"latency_p99": 200})

        # Average should be 150
        assert pattern.average_performance["latency_p99"] == pytest.approx(150, abs=1)

    def test_add_duplicate_instance(self):
        """Test that duplicate instances are not added."""
        pattern = Pattern(id="p1", name="Test", description="Test", template={})

        pattern.add_instance("upir-1")
        pattern.add_instance("upir-1")  # Duplicate

        assert len(pattern.instances) == 1


class TestSimilarityMatching:
    """Tests for pattern similarity matching."""

    def test_matches_with_centroid(self):
        """Test matching with centroid."""
        pattern = Pattern(
            id="p1",
            name="Test",
            description="Test",
            template={"centroid": [1.0, 0.0, 0.0]}
        )

        # Similar vector
        vector = [0.9, 0.1, 0.0]
        assert pattern.matches(vector, threshold=0.8)

    def test_no_match_different_vector(self):
        """Test that different vectors don't match."""
        pattern = Pattern(
            id="p1",
            name="Test",
            description="Test",
            template={"centroid": [1.0, 0.0, 0.0]}
        )

        # Orthogonal vector
        vector = [0.0, 1.0, 0.0]
        assert not pattern.matches(vector, threshold=0.8)

    def test_matches_without_centroid(self):
        """Test matching returns False when no centroid."""
        pattern = Pattern(
            id="p1",
            name="Test",
            description="Test",
            template={}  # No centroid
        )

        vector = [1.0, 0.0, 0.0]
        assert not pattern.matches(vector)

    def test_matches_custom_threshold(self):
        """Test matching with custom threshold."""
        pattern = Pattern(
            id="p1",
            name="Test",
            description="Test",
            template={"centroid": [1.0, 0.0, 0.0]}
        )

        vector = [0.7, 0.7, 0.0]  # ~0.71 similarity

        # Should match with low threshold
        assert pattern.matches(vector, threshold=0.5)

        # Should not match with high threshold
        assert not pattern.matches(vector, threshold=0.9)


class TestSerialization:
    """Tests for pattern serialization."""

    def test_to_dict(self):
        """Test converting pattern to dictionary."""
        pattern = Pattern(
            id="p1",
            name="Test",
            description="Test",
            template={"components": []},
            instances=["upir-1"],
            success_rate=0.8
        )

        d = pattern.to_dict()

        assert d["id"] == "p1"
        assert d["name"] == "Test"
        assert d["instances"] == ["upir-1"]
        assert d["success_rate"] == 0.8

    def test_from_dict(self):
        """Test creating pattern from dictionary."""
        data = {
            "id": "p1",
            "name": "Test",
            "description": "Test pattern",
            "template": {"components": []},
            "instances": ["upir-1", "upir-2"],
            "success_rate": 0.9,
            "average_performance": {"latency_p99": 100}
        }

        pattern = Pattern.from_dict(data)

        assert pattern.id == "p1"
        assert pattern.name == "Test"
        assert len(pattern.instances) == 2
        assert pattern.success_rate == 0.9
        assert pattern.average_performance["latency_p99"] == 100

    def test_from_dict_minimal(self):
        """Test from_dict with minimal fields."""
        data = {
            "id": "p1",
            "name": "Test",
            "description": "Test"
        }

        pattern = Pattern.from_dict(data)

        assert pattern.id == "p1"
        assert len(pattern.instances) == 0
        assert pattern.success_rate == 0.0

    def test_round_trip(self):
        """Test that to_dict -> from_dict preserves data."""
        original = Pattern(
            id="p1",
            name="Test",
            description="Test",
            template={"components": [{"type": "api"}]},
            instances=["u1", "u2"],
            success_rate=0.75
        )

        data = original.to_dict()
        restored = Pattern.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.instances == original.instances
        assert restored.success_rate == original.success_rate


class TestStringRepresentations:
    """Tests for string representations."""

    def test_str(self):
        """Test string representation."""
        pattern = Pattern(
            id="p1",
            name="Streaming ETL",
            description="Test",
            template={},
            instances=["u1", "u2"],
            success_rate=0.85
        )

        s = str(pattern)
        assert "Streaming ETL" in s
        assert "2 instances" in s
        assert "0.85" in s

    def test_repr(self):
        """Test repr representation."""
        pattern = Pattern(
            id="p1",
            name="Test",
            description="Test",
            template={}
        )

        r = repr(pattern)
        assert "Pattern" in r
        assert "id='p1'" in r
