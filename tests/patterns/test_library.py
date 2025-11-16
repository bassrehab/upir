"""
Unit tests for PatternLibrary.

Tests verify:
- Pattern storage and retrieval
- Search functionality
- Architecture matching
- Success rate updates
- Persistence (save/load)
- Built-in patterns

Author: Subhadip Mitra
License: Apache 2.0
"""

import tempfile
from pathlib import Path

from upir.core.architecture import Architecture
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.core.upir import UPIR
from upir.patterns.library import PatternLibrary
from upir.patterns.pattern import Pattern


class TestLibraryCreation:
    """Tests for PatternLibrary creation."""

    def test_create_default(self):
        """Test creating library with default storage path."""
        library = PatternLibrary()
        assert library.storage_path == Path("patterns.json")
        # Should have built-in patterns
        assert len(library) >= 10

    def test_create_custom_path(self):
        """Test creating library with custom storage path."""
        library = PatternLibrary(storage_path="custom.json")
        assert library.storage_path == Path("custom.json")

    def test_builtin_patterns_loaded(self):
        """Test that built-in patterns are loaded on init."""
        library = PatternLibrary()

        # Check for expected built-in patterns
        expected_patterns = [
            "streaming-etl",
            "batch-processing",
            "api-gateway",
            "event-driven",
            "lambda-architecture",
            "kappa-architecture",
            "cqrs",
            "event-sourcing",
            "pubsub-fanout",
            "mapreduce",
        ]

        for pattern_id in expected_patterns:
            assert pattern_id in library
            pattern = library.get_pattern(pattern_id)
            assert pattern is not None
            assert pattern.id == pattern_id

    def test_str_repr(self):
        """Test string representations."""
        library = PatternLibrary()
        assert "PatternLibrary" in str(library)
        assert "patterns" in str(library).lower()
        assert "custom.json" in repr(PatternLibrary("custom.json"))


class TestAddGetPattern:
    """Tests for adding and retrieving patterns."""

    def test_add_pattern(self):
        """Test adding a pattern."""
        library = PatternLibrary()
        initial_count = len(library)

        pattern = Pattern(
            id="custom-1",
            name="Custom Pattern",
            description="Test pattern",
            template={"components": []},
        )

        pattern_id = library.add_pattern(pattern)

        assert pattern_id == "custom-1"
        assert len(library) == initial_count + 1
        assert "custom-1" in library

    def test_get_existing_pattern(self):
        """Test retrieving existing pattern."""
        library = PatternLibrary()

        pattern = library.get_pattern("streaming-etl")

        assert pattern is not None
        assert pattern.id == "streaming-etl"
        assert pattern.name == "Streaming ETL Pipeline"

    def test_get_nonexistent_pattern(self):
        """Test retrieving non-existent pattern."""
        library = PatternLibrary()

        pattern = library.get_pattern("nonexistent")

        assert pattern is None

    def test_overwrite_pattern(self):
        """Test overwriting existing pattern."""
        library = PatternLibrary()

        original = library.get_pattern("streaming-etl")
        original_name = original.name

        # Create new pattern with same ID
        new_pattern = Pattern(
            id="streaming-etl",
            name="Modified Streaming ETL",
            description="Modified",
            template={"components": []},
        )

        library.add_pattern(new_pattern)

        updated = library.get_pattern("streaming-etl")
        assert updated.name == "Modified Streaming ETL"
        assert updated.name != original_name


class TestSearchPatterns:
    """Tests for pattern search."""

    def test_search_by_component_types(self):
        """Test searching by component types."""
        library = PatternLibrary()

        results = library.search_patterns(
            {"component_types": ["streaming_processor"]}
        )

        # Should find streaming patterns
        assert len(results) > 0
        for pattern in results:
            component_types = [
                comp.get("type") for comp in pattern.template.get("components", [])
            ]
            assert "streaming_processor" in component_types

    def test_search_by_min_success_rate(self):
        """Test searching by minimum success rate."""
        library = PatternLibrary()

        results = library.search_patterns({"min_success_rate": 0.85})

        # Should only find high-success patterns
        assert len(results) > 0
        for pattern in results:
            assert pattern.success_rate >= 0.85

    def test_search_by_name_contains(self):
        """Test searching by name substring."""
        library = PatternLibrary()

        results = library.search_patterns({"name_contains": "streaming"})

        # Should find patterns with "streaming" in name
        assert len(results) > 0
        for pattern in results:
            assert "streaming" in pattern.name.lower()

    def test_search_multiple_criteria(self):
        """Test searching with multiple criteria."""
        library = PatternLibrary()

        results = library.search_patterns(
            {
                "component_types": ["pubsub"],
                "min_success_rate": 0.80,
            }
        )

        # Should satisfy all criteria
        for pattern in results:
            assert pattern.success_rate >= 0.80
            component_types = [
                comp.get("type") for comp in pattern.template.get("components", [])
            ]
            assert "pubsub" in component_types

    def test_search_no_matches(self):
        """Test search with no matches."""
        library = PatternLibrary()

        results = library.search_patterns(
            {"component_types": ["nonexistent_type"]}
        )

        assert len(results) == 0

    def test_search_empty_query(self):
        """Test search with empty query returns all patterns."""
        library = PatternLibrary()

        results = library.search_patterns({})

        # Should return all patterns
        assert len(results) == len(library)


class TestMatchArchitecture:
    """Tests for architecture matching."""

    def test_match_streaming_architecture(self):
        """Test matching a streaming architecture."""
        library = PatternLibrary()

        # Create streaming UPIR
        comp1 = {"id": "source", "name": "Source", "type": "pubsub_source"}
        comp2 = {"id": "proc", "name": "Processor", "type": "streaming_processor"}
        comp3 = {"id": "db", "name": "Database", "type": "database"}
        arch = Architecture(
            components=[comp1, comp2, comp3],
            connections=[{"from": "source", "to": "proc"}, {"from": "proc", "to": "db"}],
        )
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        matches = library.match_architecture(upir, threshold=0.5)

        # Should match streaming-etl pattern
        assert len(matches) > 0
        pattern_ids = [pattern.id for pattern, _ in matches]
        assert "streaming-etl" in pattern_ids

    def test_match_batch_architecture(self):
        """Test matching a batch architecture."""
        library = PatternLibrary()

        # Create batch UPIR
        comp1 = {"id": "source", "name": "Source", "type": "bigquery_source"}
        comp2 = {"id": "proc", "name": "Processor", "type": "batch_processor"}
        comp3 = {"id": "db", "name": "Database", "type": "database"}
        arch = Architecture(
            components=[comp1, comp2, comp3],
            connections=[{"from": "source", "to": "proc"}, {"from": "proc", "to": "db"}],
        )
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        matches = library.match_architecture(upir, threshold=0.5)

        # Should match batch-processing pattern
        assert len(matches) > 0
        pattern_ids = [pattern.id for pattern, _ in matches]
        assert "batch-processing" in pattern_ids

    def test_match_api_architecture(self):
        """Test matching an API architecture."""
        library = PatternLibrary()

        # Create API UPIR
        comp1 = {"id": "gw", "name": "Gateway", "type": "api_gateway"}
        comp2 = {"id": "api", "name": "API", "type": "api_server"}
        comp3 = {"id": "db", "name": "Database", "type": "database"}
        arch = Architecture(
            components=[comp1, comp2, comp3],
            connections=[{"from": "gw", "to": "api"}, {"from": "api", "to": "db"}],
        )
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        matches = library.match_architecture(upir, threshold=0.5)

        # Should match api-gateway pattern
        assert len(matches) > 0
        pattern_ids = [pattern.id for pattern, _ in matches]
        assert "api-gateway" in pattern_ids

    def test_match_sorted_by_similarity(self):
        """Test that matches are sorted by similarity."""
        library = PatternLibrary()

        # Create streaming UPIR
        comp1 = {"id": "source", "name": "Source", "type": "pubsub_source"}
        comp2 = {"id": "proc", "name": "Processor", "type": "streaming_processor"}
        arch = Architecture(
            components=[comp1, comp2], connections=[{"from": "source", "to": "proc"}]
        )
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        matches = library.match_architecture(upir, threshold=0.3)

        # Check sorted by similarity (descending)
        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert matches[i][1] >= matches[i + 1][1]

    def test_match_with_high_threshold(self):
        """Test matching with high threshold."""
        library = PatternLibrary()

        # Create simple UPIR
        comp = {"id": "c1", "name": "C1", "type": "processor"}
        arch = Architecture(components=[comp], connections=[])
        upir = UPIR(id="test", name="Test", description="Test", architecture=arch)

        matches = library.match_architecture(upir, threshold=0.95)

        # Should have few or no matches
        assert len(matches) < 5

    def test_match_empty_architecture(self):
        """Test matching UPIR with no architecture."""
        library = PatternLibrary()

        upir = UPIR(id="test", name="Test", description="Test")

        matches = library.match_architecture(upir, threshold=0.5)

        # Should still work, may have no matches
        assert isinstance(matches, list)

    def test_match_with_specification(self):
        """Test matching architecture with specification."""
        library = PatternLibrary()

        # Create UPIR with latency constraint
        comp1 = {"id": "api", "name": "API", "type": "api_server"}
        comp2 = {"id": "db", "name": "DB", "type": "database"}
        arch = Architecture(
            components=[comp1, comp2], connections=[{"from": "api", "to": "db"}]
        )

        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.WITHIN,
                    predicate="respond",
                    time_bound=50,  # Low latency
                )
            ]
        )

        upir = UPIR(
            id="test",
            name="Test",
            description="Test",
            architecture=arch,
            specification=spec,
        )

        matches = library.match_architecture(upir, threshold=0.5)

        # Should match API pattern (has low latency)
        assert len(matches) > 0


class TestUpdateSuccessRate:
    """Tests for success rate updates."""

    def test_update_success_true(self):
        """Test updating with success."""
        library = PatternLibrary()

        pattern = library.get_pattern("streaming-etl")
        initial_rate = pattern.success_rate

        library.update_success_rate("streaming-etl", success=True)

        updated_rate = library.get_pattern("streaming-etl").success_rate

        # Success rate should increase slightly
        # With informative prior (10 pseudo-obs) and high initial rate (0.85),
        # one success has small effect: 9.5/11 ≈ 0.863
        assert updated_rate > initial_rate
        assert updated_rate < 1.0

    def test_update_success_false(self):
        """Test updating with failure."""
        library = PatternLibrary()

        pattern = library.get_pattern("streaming-etl")
        initial_rate = pattern.success_rate

        library.update_success_rate("streaming-etl", success=False)

        updated_rate = library.get_pattern("streaming-etl").success_rate

        # Success rate should decrease
        assert updated_rate < initial_rate

    def test_update_nonexistent_pattern(self):
        """Test updating non-existent pattern (should not crash)."""
        library = PatternLibrary()

        # Should not raise exception
        library.update_success_rate("nonexistent", success=True)

    def test_bayesian_update(self):
        """Test that Bayesian updating works correctly."""
        library = PatternLibrary()

        # Create pattern with known success rate
        pattern = Pattern(
            id="test-pattern",
            name="Test",
            description="Test",
            template={"components": []},
            success_rate=0.5,
            instances=["u1", "u2"],  # 2 instances
        )
        library.add_pattern(pattern)

        # Update with success
        library.update_success_rate("test-pattern", success=True)

        updated_rate = library.get_pattern("test-pattern").success_rate

        # With Beta(α, β): prior is (2, 2) inferred from success_rate=0.5
        # After success: Beta(3, 2) → mean = 3/5 = 0.6
        assert 0.55 < updated_rate < 0.65


class TestPersistence:
    """Tests for save/load functionality."""

    def test_save_and_load(self):
        """Test saving and loading patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "patterns.json"

            # Create library with custom pattern
            library1 = PatternLibrary(storage_path=str(storage_path))
            custom = Pattern(
                id="custom",
                name="Custom",
                description="Custom pattern",
                template={"components": []},
            )
            library1.add_pattern(custom)

            # Save
            library1.save()

            # Load in new library
            library2 = PatternLibrary(storage_path=str(storage_path))
            library2.load()

            # Should have loaded the custom pattern
            assert "custom" in library2
            loaded = library2.get_pattern("custom")
            assert loaded.name == "Custom"

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "nonexistent.json"

            library = PatternLibrary(storage_path=str(storage_path))
            library.load()  # Should not crash

            # Should still have built-in patterns
            assert len(library) >= 10

    def test_load_corrupted_file(self):
        """Test loading from corrupted file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "corrupted.json"

            # Write invalid JSON
            with open(storage_path, "w") as f:
                f.write("invalid json {[")

            library = PatternLibrary(storage_path=str(storage_path))
            library.load()  # Should not crash

            # Should fall back to built-in patterns
            assert len(library) >= 10

    def test_save_preserves_all_fields(self):
        """Test that save preserves all pattern fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "patterns.json"

            # Create pattern with all fields
            library1 = PatternLibrary(storage_path=str(storage_path))
            pattern = Pattern(
                id="complete",
                name="Complete Pattern",
                description="Has all fields",
                template={"components": [{"type": "api"}], "centroid": [1.0, 0.5]},
                instances=["u1", "u2", "u3"],
                success_rate=0.85,
                average_performance={"latency_p99": 50},
            )
            library1.add_pattern(pattern)
            library1.save()

            # Load and verify
            library2 = PatternLibrary(storage_path=str(storage_path))
            library2.load()

            loaded = library2.get_pattern("complete")
            assert loaded.name == "Complete Pattern"
            assert len(loaded.instances) == 3
            assert loaded.success_rate == 0.85
            assert loaded.average_performance["latency_p99"] == 50


class TestBuiltinPatterns:
    """Tests for built-in patterns."""

    def test_streaming_etl_pattern(self):
        """Test streaming ETL pattern structure."""
        library = PatternLibrary()
        pattern = library.get_pattern("streaming-etl")

        assert pattern.name == "Streaming ETL Pipeline"
        assert pattern.success_rate > 0.8
        assert "centroid" in pattern.template
        assert len(pattern.template["centroid"]) == 32

    def test_batch_processing_pattern(self):
        """Test batch processing pattern."""
        library = PatternLibrary()
        pattern = library.get_pattern("batch-processing")

        assert pattern.name == "Batch Processing Pipeline"
        assert "batch_processor" in str(pattern.template)

    def test_api_gateway_pattern(self):
        """Test API gateway pattern."""
        library = PatternLibrary()
        pattern = library.get_pattern("api-gateway")

        assert pattern.name == "Request-Response API"
        assert "api" in pattern.name.lower()

    def test_all_builtin_patterns_valid(self):
        """Test that all built-in patterns are valid."""
        library = PatternLibrary()

        builtin_ids = [
            "streaming-etl",
            "batch-processing",
            "api-gateway",
            "event-driven",
            "lambda-architecture",
            "kappa-architecture",
            "cqrs",
            "event-sourcing",
            "pubsub-fanout",
            "mapreduce",
        ]

        for pattern_id in builtin_ids:
            pattern = library.get_pattern(pattern_id)

            # Validate structure
            assert pattern is not None
            assert pattern.id == pattern_id
            assert len(pattern.name) > 0
            assert len(pattern.description) > 0
            assert "components" in pattern.template
            assert "centroid" in pattern.template
            assert len(pattern.template["centroid"]) == 32
            assert 0.0 <= pattern.success_rate <= 1.0


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self):
        """Test complete pattern matching workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "patterns.json"

            # 1. Create library
            library = PatternLibrary(storage_path=str(storage_path))

            # 2. Create UPIR
            comp1 = {"id": "ps", "name": "PubSub", "type": "pubsub_source"}
            comp2 = {"id": "proc", "name": "Processor", "type": "streaming_processor"}
            comp3 = {"id": "db", "name": "Database", "type": "database"}
            arch = Architecture(
                components=[comp1, comp2, comp3],
                connections=[
                    {"from": "ps", "to": "proc"},
                    {"from": "proc", "to": "db"},
                ],
            )
            upir = UPIR(id="my-upir", name="My UPIR", description="Test", architecture=arch)

            # 3. Match architecture
            matches = library.match_architecture(upir, threshold=0.6)
            assert len(matches) > 0

            best_pattern, score = matches[0]
            assert score > 0.6

            # 4. Update success rate
            initial_rate = best_pattern.success_rate
            library.update_success_rate(best_pattern.id, success=True)
            updated_rate = library.get_pattern(best_pattern.id).success_rate
            # Should increase (Bayesian update with success)
            assert updated_rate > initial_rate

            # 5. Search for similar patterns
            results = library.search_patterns(
                {"component_types": ["streaming_processor"]}
            )
            assert len(results) > 0

            # 6. Save library
            library.save()

            # 7. Load in new library
            library2 = PatternLibrary(storage_path=str(storage_path))
            library2.load()

            # Should have same patterns
            assert len(library2) == len(library)

    def test_custom_pattern_matching(self):
        """Test adding custom pattern and matching."""
        library = PatternLibrary()

        # Add custom pattern
        from upir.patterns.extractor import PatternExtractor

        extractor = PatternExtractor()

        # Create custom pattern based on UPIR
        comp1 = {"id": "c1", "name": "C1", "type": "custom_processor"}
        comp2 = {"id": "c2", "name": "C2", "type": "custom_db"}
        arch = Architecture(
            components=[comp1, comp2], connections=[{"from": "c1", "to": "c2"}]
        )
        upir = UPIR(id="custom", name="Custom", description="Custom", architecture=arch)

        features = extractor.extract_features(upir)

        custom_pattern = Pattern(
            id="my-custom",
            name="My Custom Pattern",
            description="Custom pattern",
            template={"components": [{"type": "custom_processor"}], "centroid": features.tolist()},
            success_rate=0.9,
        )

        library.add_pattern(custom_pattern)

        # Match against similar UPIR
        comp1 = {"id": "c1", "name": "C1", "type": "custom_processor"}
        comp2 = {"id": "c2", "name": "C2", "type": "custom_db"}
        arch2 = Architecture(
            components=[comp1, comp2], connections=[{"from": "c1", "to": "c2"}]
        )
        upir2 = UPIR(id="test", name="Test", description="Test", architecture=arch2)

        matches = library.match_architecture(upir2, threshold=0.5)

        # Should match custom pattern highly
        pattern_ids = [p.id for p, _ in matches]
        assert "my-custom" in pattern_ids
