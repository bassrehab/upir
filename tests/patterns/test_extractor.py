"""
Unit tests for PatternExtractor.

Tests verify:
- Feature extraction from UPIRs
- Architecture clustering
- Pattern extraction from clusters
- Pattern discovery pipeline

Author: Subhadip Mitra
License: Apache 2.0
"""

import numpy as np

from upir.core.architecture import Architecture
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.core.upir import UPIR
from upir.patterns.extractor import PatternExtractor
from upir.patterns.pattern import Pattern


class TestExtractorCreation:
    """Tests for PatternExtractor creation."""

    def test_create_default(self):
        """Test creating extractor with defaults."""
        extractor = PatternExtractor()
        assert extractor.n_clusters == 10
        assert extractor.feature_dim == 32

    def test_create_custom(self):
        """Test creating extractor with custom parameters."""
        extractor = PatternExtractor(n_clusters=5, feature_dim=64)
        assert extractor.n_clusters == 5
        assert extractor.feature_dim == 64

    def test_str_repr(self):
        """Test string representations."""
        extractor = PatternExtractor(n_clusters=5)
        assert "PatternExtractor" in str(extractor)
        assert "n_clusters=5" in repr(extractor)


class TestFeatureExtraction:
    """Tests for feature extraction."""

    def test_extract_features_empty_architecture(self):
        """Test extracting features from UPIR with no architecture."""
        extractor = PatternExtractor()
        upir = UPIR(id="test", name="Test", description="Test")

        features = extractor.extract_features(upir)

        assert features.shape == (32,)
        assert np.all(features == 0)

    def test_extract_features_simple_architecture(self):
        """Test extracting features from simple architecture."""
        extractor = PatternExtractor()

        comp1 = {"id": "c1", "name": "C1", "type": "streaming_processor"}
        comp2 = {"id": "c2", "name": "C2", "type": "database"}
        conn = {"from": "c1", "to": "c2"}

        arch = Architecture(components=[comp1, comp2], connections=[conn])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        features = extractor.extract_features(upir)

        assert features.shape == (32,)
        assert np.all(features >= 0)
        assert np.all(features <= 1)
        assert features[0] > 0  # Component count
        assert features[1] > 0  # Connection count

    def test_extract_features_with_specification(self):
        """Test feature extraction with specification."""
        extractor = PatternExtractor()

        comp = {"id": "c1", "name": "C1", "type": "api_server"}
        arch = Architecture(components=[comp], connections=[])

        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.WITHIN,
                    predicate="respond",
                    time_bound=100  # 100ms latency requirement
                )
            ]
        )

        upir = UPIR(
            id="test",
            name="Test",
            description="Test",
            architecture=arch,
            specification=spec
        )

        features = extractor.extract_features(upir)

        assert features.shape == (32,)
        assert np.any(features > 0)  # Should have non-zero features

    def test_extract_features_deterministic(self):
        """Test that feature extraction is deterministic."""
        extractor = PatternExtractor()

        comp = {"id": "c1", "name": "C1", "type": "processor"}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        features1 = extractor.extract_features(upir)
        features2 = extractor.extract_features(upir)

        assert np.allclose(features1, features2)

    def test_extract_features_normalization(self):
        """Test that features are normalized to [0, 1]."""
        extractor = PatternExtractor()

        # Create architecture with many components
        components = [
            {"id": f"c{i}", "name": f"C{i}", "type": "streaming_processor"}
            for i in range(20)
        ]
        connections = [
            {"from": f"c{i}", "to": f"c{i+1}"}
            for i in range(19)
        ]

        arch = Architecture(components=components, connections=connections)
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        features = extractor.extract_features(upir)

        # All features should be in [0, 1]
        assert np.all(features >= 0)
        assert np.all(features <= 1)


class TestClusterArchitectures:
    """Tests for architecture clustering."""

    def test_cluster_simple_upirs(self):
        """Test clustering simple UPIRs."""
        extractor = PatternExtractor(n_clusters=2)

        # Create streaming UPIRs
        streaming_upirs = []
        for i in range(3):
            comp = {"id": f"c{i}", "name": f"C{i}", "type": "streaming_processor"}
            arch = Architecture(components=[comp], connections=[])
            upir = UPIR(id=f"stream-{i}", name=f"Stream {i}", description="Streaming", architecture=arch)
            streaming_upirs.append(upir)

        # Create batch UPIRs
        batch_upirs = []
        for i in range(3):
            comp = {"id": f"c{i}", "name": f"C{i}", "type": "batch_processor"}
            arch = Architecture(components=[comp], connections=[])
            upir = UPIR(id=f"batch-{i}", name=f"Batch {i}", description="Batch", architecture=arch)
            batch_upirs.append(upir)

        all_upirs = streaming_upirs + batch_upirs

        clusters = extractor.cluster_architectures(all_upirs)

        assert len(clusters) == 2
        # Each cluster should have some UPIRs
        assert all(len(cluster) > 0 for cluster in clusters.values())

    def test_cluster_adjusts_to_upir_count(self):
        """Test that clustering adjusts when fewer UPIRs than clusters."""
        extractor = PatternExtractor(n_clusters=10)

        # Only 3 UPIRs
        upirs = []
        for i in range(3):
            comp = {"id": f"c{i}", "name": f"C{i}", "type": "processor"}
            arch = Architecture(components=[comp], connections=[])
            upir = UPIR(id=f"u{i}", name=f"U{i}", description="Test", architecture=arch)
            upirs.append(upir)

        clusters = extractor.cluster_architectures(upirs)

        # Should create 3 clusters, not 10
        assert len(clusters) == 3

    def test_cluster_empty_list(self):
        """Test clustering with empty list."""
        extractor = PatternExtractor(n_clusters=5)

        clusters = extractor.cluster_architectures([])

        assert len(clusters) == 0


class TestExtractPattern:
    """Tests for pattern extraction from clusters."""

    def test_extract_pattern_from_cluster(self):
        """Test extracting pattern from cluster."""
        extractor = PatternExtractor()

        # Create cluster of similar UPIRs
        cluster = []
        for i in range(5):
            comp = {"id": f"c{i}", "name": f"C{i}", "type": "api_server"}
            arch = Architecture(components=[comp], connections=[])
            upir = UPIR(id=f"u{i}", name=f"U{i}", description="API", architecture=arch)
            cluster.append(upir)

        pattern = extractor.extract_pattern(cluster, cluster_id=0)

        assert isinstance(pattern, Pattern)
        assert pattern.id == "pattern-0"
        assert len(pattern.instances) == 5
        assert "components" in pattern.template
        assert "parameters" in pattern.template
        assert "centroid" in pattern.template

    def test_extract_pattern_empty_cluster(self):
        """Test extracting pattern from empty cluster."""
        extractor = PatternExtractor()

        pattern = extractor.extract_pattern([], cluster_id=0)

        assert pattern.id == "pattern-0"
        assert len(pattern.instances) == 0

    def test_extract_pattern_includes_centroid(self):
        """Test that extracted pattern includes centroid."""
        extractor = PatternExtractor()

        cluster = []
        for i in range(3):
            comp = {"id": f"c{i}", "name": f"C{i}", "type": "processor"}
            arch = Architecture(components=[comp], connections=[])
            upir = UPIR(id=f"u{i}", name=f"U{i}", description="Test", architecture=arch)
            cluster.append(upir)

        pattern = extractor.extract_pattern(cluster, cluster_id=0)

        assert "centroid" in pattern.template
        assert len(pattern.template["centroid"]) == extractor.feature_dim


class TestDiscoverPatterns:
    """Tests for pattern discovery pipeline."""

    def test_discover_patterns_basic(self):
        """Test discovering patterns from UPIRs."""
        extractor = PatternExtractor(n_clusters=2)

        # Create diverse UPIRs
        upirs = []
        for i in range(6):
            comp_type = "streaming_processor" if i < 3 else "batch_processor"
            comp = {"id": f"c{i}", "name": f"C{i}", "type": comp_type}
            arch = Architecture(components=[comp], connections=[])
            upir = UPIR(id=f"u{i}", name=f"U{i}", description="Test", architecture=arch)
            upirs.append(upir)

        patterns = extractor.discover_patterns(upirs)

        assert len(patterns) > 0
        assert all(isinstance(p, Pattern) for p in patterns)
        # Total instances across patterns should equal total UPIRs
        total_instances = sum(len(p.instances) for p in patterns)
        assert total_instances == len(upirs)

    def test_discover_patterns_empty_list(self):
        """Test discovering patterns from empty list."""
        extractor = PatternExtractor(n_clusters=3)

        patterns = extractor.discover_patterns([])

        assert len(patterns) == 0

    def test_discover_patterns_sorts_by_size(self):
        """Test that patterns are sorted by instance count."""
        extractor = PatternExtractor(n_clusters=3)

        # Create UPIRs with skewed distribution
        upirs = []
        # 6 streaming UPIRs
        for i in range(6):
            comp = {"id": f"c{i}", "name": f"C{i}", "type": "streaming_processor"}
            arch = Architecture(components=[comp], connections=[])
            upir = UPIR(id=f"stream-{i}", name=f"S{i}", description="Stream", architecture=arch)
            upirs.append(upir)

        # 2 batch UPIRs
        for i in range(2):
            comp = {"id": f"c{i}", "name": f"C{i}", "type": "batch_processor"}
            arch = Architecture(components=[comp], connections=[])
            upir = UPIR(id=f"batch-{i}", name=f"B{i}", description="Batch", architecture=arch)
            upirs.append(upir)

        patterns = extractor.discover_patterns(upirs)

        # Patterns should be sorted by instance count (descending)
        for i in range(len(patterns) - 1):
            assert len(patterns[i].instances) >= len(patterns[i + 1].instances)


class TestClassifyUpir:
    """Tests for UPIR classification."""

    def test_classify_upir(self):
        """Test classifying a UPIR into a cluster."""
        extractor = PatternExtractor(n_clusters=2)

        # Train on some UPIRs
        training_upirs = []
        for i in range(4):
            comp = {"id": f"c{i}", "name": f"C{i}", "type": "streaming_processor"}
            arch = Architecture(components=[comp], connections=[])
            upir = UPIR(id=f"u{i}", name=f"U{i}", description="Test", architecture=arch)
            training_upirs.append(upir)

        extractor.discover_patterns(training_upirs)

        # Classify new UPIR
        new_comp = {"id": "c_new", "name": "New", "type": "streaming_processor"}
        new_arch = Architecture(components=[new_comp], connections=[])
        new_upir = UPIR(id="new", name="New", description="Test", architecture=new_arch)

        cluster_id = extractor.classify_upir(new_upir)

        assert isinstance(cluster_id, (int, np.integer))
        assert 0 <= cluster_id < extractor.n_clusters


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline(self):
        """Test full pattern discovery and classification pipeline."""
        extractor = PatternExtractor(n_clusters=3)

        # Create training set with different patterns
        upirs = []

        # Streaming pattern (4 instances)
        for i in range(4):
            comp1 = {"id": f"source-{i}", "name": f"Source{i}", "type": "pubsub_source"}
            comp2 = {"id": f"proc-{i}", "name": f"Proc{i}", "type": "streaming_processor"}
            conn = {"from": f"source-{i}", "to": f"proc-{i}"}
            arch = Architecture(components=[comp1, comp2], connections=[conn])

            spec = FormalSpecification(
                properties=[
                    TemporalProperty(
                        operator=TemporalOperator.WITHIN,
                        predicate="process",
                        time_bound=100
                    )
                ]
            )

            upir = UPIR(
                id=f"stream-{i}",
                name=f"Streaming {i}",
                description="Streaming",
                architecture=arch,
                specification=spec
            )
            upirs.append(upir)

        # Batch pattern (3 instances)
        for i in range(3):
            comp1 = {"id": f"source-{i}", "name": f"Source{i}", "type": "bigquery_source"}
            comp2 = {"id": f"proc-{i}", "name": f"Proc{i}", "type": "batch_processor"}
            conn = {"from": f"source-{i}", "to": f"proc-{i}"}
            arch = Architecture(components=[comp1, comp2], connections=[conn])

            upir = UPIR(
                id=f"batch-{i}",
                name=f"Batch {i}",
                description="Batch",
                architecture=arch
            )
            upirs.append(upir)

        # Discover patterns
        patterns = extractor.discover_patterns(upirs)

        assert len(patterns) > 0

        # Classify a new streaming UPIR
        new_comp1 = {"id": "new_source", "name": "NewSource", "type": "pubsub_source"}
        new_comp2 = {"id": "new_proc", "name": "NewProc", "type": "streaming_processor"}
        new_conn = {"from": "new_source", "to": "new_proc"}
        new_arch = Architecture(components=[new_comp1, new_comp2], connections=[new_conn])
        new_upir = UPIR(id="new", name="New", description="New", architecture=new_arch)

        cluster_id = extractor.classify_upir(new_upir)

        # Should be classified into a cluster
        assert 0 <= cluster_id < extractor.n_clusters
