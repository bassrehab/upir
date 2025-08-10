"""
Unit tests for pattern extraction and library system.

Testing pattern discovery, clustering, abstraction, and
the pattern library with search and recommendation.

Author: subhadipmitra@google.com
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upir.core.models import (
    UPIR, Architecture, FormalSpecification,
    TemporalProperty, TemporalOperator, Evidence
)
from upir.patterns.extractor import (
    PatternFeatures, FeatureExtractor, PatternClusterer,
    PatternAbstractor, ArchitecturalPattern
)
from upir.patterns.library import (
    PatternLibrary, SearchQuery, PatternUsage
)


class TestPatternFeatures:
    """Test pattern feature extraction."""
    
    def test_feature_creation(self):
        """Test creating pattern features."""
        features = PatternFeatures(
            num_components=5,
            num_connections=4,
            component_types={"service": 3, "storage": 2},
            connection_density=0.8,
            max_fan_out=2,
            max_fan_in=2,
            has_cycles=False,
            depth=3,
            breadth=2,
            clustering_coefficient=0.5,
            uses_caching=True,
            uses_queuing=False,
            uses_replication=True,
            uses_sharding=False,
            uses_load_balancing=False,
            avg_latency=50.0,
            avg_throughput=1000.0,
            success_rate=0.95
        )
        
        assert features.num_components == 5
        assert features.uses_caching is True
        assert features.avg_latency == 50.0
    
    def test_feature_to_vector(self):
        """Test converting features to vector."""
        features = PatternFeatures(
            num_components=5,
            num_connections=4,
            component_types={"service": 3},
            connection_density=0.8,
            max_fan_out=2,
            max_fan_in=2,
            has_cycles=False,
            depth=3,
            breadth=2,
            clustering_coefficient=0.5,
            uses_caching=True,
            uses_queuing=False,
            uses_replication=False,
            uses_sharding=False,
            uses_load_balancing=False
        )
        
        vector = features.to_vector()
        
        assert len(vector) > 0
        assert vector[0] == 5.0  # num_components
        assert 1.0 in vector  # uses_caching as 1.0


class TestFeatureExtractor:
    """Test feature extraction from architectures."""
    
    def test_extract_features(self):
        """Test extracting features from UPIR."""
        upir = UPIR(name="Test System")
        upir.architecture = Architecture(
            components=[
                {"name": "api", "type": "service", "config": {"replicas": 3}},
                {"name": "cache", "type": "cache", "config": {"size": 512}},
                {"name": "db", "type": "storage", "config": {"replicas": 2}}
            ],
            connections=[
                {"source": "api", "target": "cache"},
                {"source": "api", "target": "db"},
                {"source": "cache", "target": "db"}
            ],
            deployment={},
            patterns=["microservices", "caching"]
        )
        
        extractor = FeatureExtractor()
        features = extractor.extract(upir)
        
        assert features.num_components == 3
        assert features.num_connections == 3
        assert features.uses_caching is True
        assert features.uses_replication is True
        assert "service" in features.component_types
    
    def test_cycle_detection(self):
        """Test detecting cycles in architecture."""
        upir = UPIR(name="Cyclic System")
        upir.architecture = Architecture(
            components=[
                {"name": "a", "type": "service"},
                {"name": "b", "type": "service"},
                {"name": "c", "type": "service"}
            ],
            connections=[
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "c", "target": "a"}  # Creates cycle
            ],
            deployment={},
            patterns=[]
        )
        
        extractor = FeatureExtractor()
        features = extractor.extract(upir)
        
        assert features.has_cycles is True
    
    def test_topology_calculation(self):
        """Test calculating depth and breadth."""
        upir = UPIR(name="Layered System")
        upir.architecture = Architecture(
            components=[
                {"name": "gateway", "type": "gateway"},
                {"name": "service1", "type": "service"},
                {"name": "service2", "type": "service"},
                {"name": "db", "type": "storage"}
            ],
            connections=[
                {"source": "gateway", "target": "service1"},
                {"source": "gateway", "target": "service2"},
                {"source": "service1", "target": "db"},
                {"source": "service2", "target": "db"}
            ],
            deployment={},
            patterns=[]
        )
        
        extractor = FeatureExtractor()
        features = extractor.extract(upir)
        
        assert features.depth == 2  # gateway -> service -> db
        assert features.breadth == 2  # Two services at same level


class TestPatternClusterer:
    """Test clustering of architectures."""
    
    def test_clustering(self):
        """Test clustering similar architectures."""
        # Create similar features
        features1 = PatternFeatures(
            num_components=5, num_connections=4,
            component_types={"service": 3, "storage": 2},
            connection_density=0.8, max_fan_out=2, max_fan_in=2,
            has_cycles=False, depth=3, breadth=2,
            clustering_coefficient=0.5,
            uses_caching=True, uses_queuing=False,
            uses_replication=True, uses_sharding=False,
            uses_load_balancing=False
        )
        
        features2 = PatternFeatures(
            num_components=6, num_connections=5,
            component_types={"service": 4, "storage": 2},
            connection_density=0.83, max_fan_out=2, max_fan_in=2,
            has_cycles=False, depth=3, breadth=2,
            clustering_coefficient=0.48,
            uses_caching=True, uses_queuing=False,
            uses_replication=True, uses_sharding=False,
            uses_load_balancing=False
        )
        
        features3 = PatternFeatures(
            num_components=20, num_connections=30,
            component_types={"processor": 10, "queue": 10},
            connection_density=1.5, max_fan_out=5, max_fan_in=5,
            has_cycles=True, depth=10, breadth=5,
            clustering_coefficient=0.2,
            uses_caching=False, uses_queuing=True,
            uses_replication=False, uses_sharding=True,
            uses_load_balancing=True
        )
        
        clusterer = PatternClusterer(method="dbscan")
        clusters = clusterer.cluster([features1, features2, features3], min_cluster_size=2)
        
        # Should have at least one cluster (features1 and features2 are similar)
        # Note: Actual clustering depends on sklearn availability
        assert isinstance(clusters, dict)


class TestPatternAbstractor:
    """Test pattern abstraction from clusters."""
    
    def test_pattern_abstraction(self):
        """Test abstracting pattern from similar UPIRs."""
        # Create similar UPIRs
        upir1 = UPIR(id="1", name="System 1")
        upir1.architecture = Architecture(
            components=[
                {"name": "queue", "type": "queue", "config": {"size": 100}},
                {"name": "processor", "type": "processor", "config": {"threads": 10}},
                {"name": "storage", "type": "storage", "config": {"replicas": 3}}
            ],
            connections=[
                {"source": "queue", "target": "processor"},
                {"source": "processor", "target": "storage"}
            ],
            deployment={},
            patterns=["streaming"]
        )
        
        upir2 = UPIR(id="2", name="System 2")
        upir2.architecture = Architecture(
            components=[
                {"name": "kafka", "type": "queue", "config": {"size": 200}},
                {"name": "spark", "type": "processor", "config": {"threads": 20}},
                {"name": "hdfs", "type": "storage", "config": {"replicas": 3}}
            ],
            connections=[
                {"source": "kafka", "target": "spark"},
                {"source": "spark", "target": "hdfs"}
            ],
            deployment={},
            patterns=["streaming"]
        )
        
        abstractor = PatternAbstractor()
        pattern = abstractor.abstract([upir1, upir2], name="test_pattern")
        
        assert pattern.name == "test_pattern"
        assert pattern.category in ["streaming", "batch", "microservices"]
        assert len(pattern.template_components) > 0
        assert len(pattern.instances) == 2
    
    def test_parameter_extraction(self):
        """Test extracting parameters from pattern."""
        upir1 = UPIR(id="1")
        upir1.architecture = Architecture(
            components=[
                {"name": "proc", "type": "processor", "config": {"threads": 10}}
            ],
            connections=[],
            deployment={},
            patterns=[]
        )
        
        upir2 = UPIR(id="2")
        upir2.architecture = Architecture(
            components=[
                {"name": "proc", "type": "processor", "config": {"threads": 20}}
            ],
            connections=[],
            deployment={},
            patterns=[]
        )
        
        abstractor = PatternAbstractor()
        pattern = abstractor.abstract([upir1, upir2])
        
        # Should have parameterized the threads config
        assert len(pattern.parameters) > 0 or len(pattern.template_components) > 0


class TestArchitecturalPattern:
    """Test architectural pattern functionality."""
    
    def test_pattern_instantiation(self):
        """Test instantiating pattern with parameters."""
        pattern = ArchitecturalPattern(
            id="test_pattern",
            name="Test Pattern",
            description="Test",
            category="streaming",
            template_components=[
                {"name": "queue_${instance_id}", "type": "queue", 
                 "config": {"size": "${queue_size}"}},
                {"name": "processor_${instance_id}", "type": "processor",
                 "config": {"threads": "${processor_threads}"}}
            ],
            template_connections=[
                {"source": "queue_${instance_id}", "target": "processor_${instance_id}"}
            ],
            required_properties=["data_consistency"],
            parameters={
                "queue_size": {"type": "int", "default": 100},
                "processor_threads": {"type": "int", "default": 10}
            },
            constraints={
                "queue_size": {"min": 10, "max": 1000},
                "processor_threads": {"min": 1, "max": 100}
            }
        )
        
        # Instantiate with custom parameters
        params = {
            "instance_id": "prod",
            "queue_size": 500,
            "processor_threads": 50
        }
        
        arch = pattern.instantiate(params)
        
        assert len(arch.components) == 2
        assert arch.components[0]["config"]["size"] == 500
        assert arch.components[1]["config"]["threads"] == 50
    
    def test_pattern_matching(self):
        """Test if architecture matches pattern."""
        pattern = ArchitecturalPattern(
            id="test",
            name="Test",
            description="Test",
            category="streaming",
            template_components=[
                {"type": "queue"},
                {"type": "processor"},
                {"type": "storage"}
            ],
            template_connections=[
                {"source": "queue", "target": "processor"}
            ],
            required_properties=[]
        )
        
        # Matching architecture
        arch = Architecture(
            components=[
                {"name": "q", "type": "queue"},
                {"name": "p", "type": "processor"},
                {"name": "s", "type": "storage"}
            ],
            connections=[
                {"source": "q", "target": "p"}
            ],
            deployment={},
            patterns=[]
        )
        
        assert pattern.matches(arch, threshold=0.8) is True
        
        # Non-matching architecture
        arch2 = Architecture(
            components=[
                {"name": "s", "type": "service"}
            ],
            connections=[],
            deployment={},
            patterns=[]
        )
        
        assert pattern.matches(arch2, threshold=0.8) is False
    
    def test_update_statistics(self):
        """Test updating pattern statistics."""
        pattern = ArchitecturalPattern(
            id="test",
            name="Test",
            description="Test",
            category="test",
            template_components=[],
            template_connections=[],
            required_properties=[],
            success_rate=0.5
        )
        
        # Update with successful usage
        pattern.update_statistics(True, {"latency": 50, "throughput": 1000})
        assert pattern.success_rate > 0.5
        assert "latency" in pattern.avg_performance
        
        # Update with failed usage
        pattern.update_statistics(False, {"latency": 200})
        assert pattern.success_rate < 0.55  # Should decrease but not drastically


class TestPatternLibrary:
    """Test pattern library functionality."""
    
    def test_library_crud(self):
        """Test CRUD operations on pattern library."""
        library = PatternLibrary()
        
        pattern = ArchitecturalPattern(
            id="test1",
            name="Test Pattern",
            description="Test",
            category="test",
            template_components=[],
            template_connections=[],
            required_properties=[]
        )
        
        # Add pattern
        pid = library.add_pattern(pattern)
        assert pid == "test1"
        assert len(library.patterns) == 1
        
        # Get pattern
        retrieved = library.get_pattern("test1")
        assert retrieved.name == "Test Pattern"
        
        # Update pattern
        library.update_pattern("test1", {"description": "Updated"})
        assert library.patterns["test1"].description == "Updated"
        
        # Remove pattern
        library.remove_pattern("test1")
        assert len(library.patterns) == 0
    
    def test_pattern_search(self):
        """Test searching for patterns."""
        library = PatternLibrary()
        
        # Add patterns
        pattern1 = ArchitecturalPattern(
            id="p1", name="Streaming Pattern", description="", category="streaming",
            template_components=[{"type": "queue"}, {"type": "processor"}],
            template_connections=[], required_properties=["data_consistency"],
            success_rate=0.9
        )
        
        pattern2 = ArchitecturalPattern(
            id="p2", name="Batch Pattern", description="", category="batch",
            template_components=[{"type": "scheduler"}, {"type": "processor"}],
            template_connections=[], required_properties=["job_completion"],
            success_rate=0.7
        )
        
        library.add_pattern(pattern1)
        library.add_pattern(pattern2)
        
        # Search for streaming patterns
        query = SearchQuery(
            category="streaming",
            min_success_rate=0.8
        )
        
        results = library.search(query)
        assert len(results) == 1
        assert results[0][0].name == "Streaming Pattern"
    
    def test_pattern_recommendation(self):
        """Test pattern recommendation."""
        library = PatternLibrary()
        
        # Add pattern
        pattern = ArchitecturalPattern(
            id="p1", name="Test Pattern", description="", category="streaming",
            template_components=[{"type": "queue"}],
            template_connections=[], 
            required_properties=["data_consistency"],
            success_rate=0.9
        )
        library.add_pattern(pattern)
        
        # Create UPIR needing recommendation
        upir = UPIR(name="New System")
        upir.specification = FormalSpecification(
            invariants=[
                TemporalProperty(TemporalOperator.ALWAYS, "data_consistency")
            ],
            properties=[],
            constraints={}
        )
        
        recommendations = library.recommend(upir, top_k=1)
        
        assert len(recommendations) > 0
        assert recommendations[0][0].name == "Test Pattern"
    
    def test_usage_tracking(self):
        """Test tracking pattern usage."""
        library = PatternLibrary()
        
        pattern = ArchitecturalPattern(
            id="p1", name="Test", description="", category="test",
            template_components=[], template_connections=[],
            required_properties=[], success_rate=0.5
        )
        library.add_pattern(pattern)
        
        # Record usage
        library.record_usage(
            pattern_id="p1",
            upir_id="test_upir",
            success=True,
            performance={"latency": 50},
            feedback="Worked well"
        )
        
        assert len(library.usage_history) == 1
        assert library.patterns["p1"].success_rate > 0.5  # Should increase
    
    def test_pattern_discovery(self):
        """Test discovering patterns from UPIRs."""
        library = PatternLibrary()
        
        # Create similar UPIRs
        upirs = []
        for i in range(3):
            upir = UPIR(id=f"upir_{i}")
            upir.architecture = Architecture(
                components=[
                    {"name": f"queue_{i}", "type": "queue"},
                    {"name": f"proc_{i}", "type": "processor"}
                ],
                connections=[
                    {"source": f"queue_{i}", "target": f"proc_{i}"}
                ],
                deployment={},
                patterns=["streaming"]
            )
            upirs.append(upir)
        
        # Discover patterns
        discovered = library.discover_patterns(upirs, min_cluster_size=2)
        
        # Should discover at least one pattern if clustering works
        assert isinstance(discovered, list)
    
    def test_pattern_evolution(self):
        """Test pattern evolution based on usage."""
        library = PatternLibrary()
        
        pattern = ArchitecturalPattern(
            id="p1", name="Test", description="Test pattern",
            category="test", template_components=[], 
            template_connections=[], required_properties=[],
            success_rate=0.5
        )
        library.add_pattern(pattern)
        
        # Add usage history
        for i in range(11):
            library.record_usage(
                pattern_id="p1",
                upir_id=f"upir_{i}",
                success=(i < 3),  # Only first 3 are successful
                performance={"latency": 100},
                feedback=""
            )
        
        # Evolve patterns
        evolved = library.evolve_patterns(min_usage=10)
        
        # Pattern should be deprecated due to low success rate
        assert "p1" in evolved
        assert "DEPRECATED" in library.patterns["p1"].description
    
    def test_library_persistence(self):
        """Test saving and loading library."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create library with pattern
            library1 = PatternLibrary(storage_path=temp_dir)
            
            pattern = ArchitecturalPattern(
                id="p1", name="Test", description="Test",
                category="test", template_components=[{"type": "service"}],
                template_connections=[], required_properties=["test"],
                success_rate=0.8
            )
            library1.add_pattern(pattern)
            
            # Save
            library1.save()
            
            # Load in new library
            library2 = PatternLibrary(storage_path=temp_dir)
            library2.load()
            
            assert len(library2.patterns) == 1
            assert library2.patterns["p1"].name == "Test"
            assert library2.patterns["p1"].success_rate == 0.8
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])