"""
Pattern Discovery and Extraction Demonstration

This example shows how UPIR discovers reusable architectural patterns
from multiple system instances using clustering and abstraction.

The demonstration:
1. Creates several similar architectures
2. Discovers patterns through clustering
3. Shows pattern abstraction and parameterization
4. Demonstrates pattern search and recommendation
5. Shows pattern evolution based on usage

Author: subhadipmitra@google.com
"""

import sys
import os
from datetime import datetime, timedelta
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upir.core.models import (
    UPIR, Architecture, FormalSpecification, 
    TemporalProperty, TemporalOperator, Evidence
)
from upir.patterns.extractor import (
    FeatureExtractor, PatternClusterer, PatternAbstractor
)
from upir.patterns.library import (
    PatternLibrary, SearchQuery
)


def create_streaming_architectures() -> list:
    """
    Create several streaming pipeline architectures with variations.
    
    These represent different implementations of similar patterns.
    """
    architectures = []
    
    # Streaming Pattern Variant 1: Kafka-based
    upir1 = UPIR(id="stream_1", name="Kafka Streaming Pipeline")
    upir1.architecture = Architecture(
        components=[
            {"name": "kafka", "type": "queue", "config": {"partitions": 10, "replication": 3}},
            {"name": "stream_processor", "type": "processor", "config": {"parallelism": 20, "window_size": 60}},
            {"name": "redis", "type": "cache", "config": {"size_mb": 1024, "ttl": 300}},
            {"name": "postgres", "type": "storage", "config": {"replicas": 3, "sharding": True}}
        ],
        connections=[
            {"source": "kafka", "target": "stream_processor"},
            {"source": "stream_processor", "target": "redis"},
            {"source": "stream_processor", "target": "postgres"},
            {"source": "redis", "target": "postgres", "condition": "cache_miss"}
        ],
        deployment={"cloud": "aws", "region": "us-east-1"},
        patterns=["streaming", "caching", "sharding"]
    )
    upir1.specification = FormalSpecification(
        invariants=[TemporalProperty(TemporalOperator.ALWAYS, "data_consistency")],
        properties=[],
        constraints={"latency": {"max": 100}, "throughput": {"min": 10000}}
    )
    # Add evidence of success
    upir1.add_evidence(Evidence("production", "production", 
                               {"latency": 50, "throughput": 15000}, confidence=0.9))
    architectures.append(upir1)
    
    # Streaming Pattern Variant 2: Pub/Sub-based
    upir2 = UPIR(id="stream_2", name="Pub/Sub Streaming Pipeline")
    upir2.architecture = Architecture(
        components=[
            {"name": "pubsub", "type": "queue", "config": {"topics": 5, "retention": 7}},
            {"name": "dataflow", "type": "processor", "config": {"parallelism": 15, "window_size": 30}},
            {"name": "memcached", "type": "cache", "config": {"size_mb": 512, "ttl": 600}},
            {"name": "bigquery", "type": "storage", "config": {"replicas": 2, "sharding": False}}
        ],
        connections=[
            {"source": "pubsub", "target": "dataflow"},
            {"source": "dataflow", "target": "memcached"},
            {"source": "dataflow", "target": "bigquery"}
        ],
        deployment={"cloud": "gcp", "region": "us-central1"},
        patterns=["streaming", "caching"]
    )
    upir2.specification = FormalSpecification(
        invariants=[TemporalProperty(TemporalOperator.ALWAYS, "data_consistency")],
        properties=[],
        constraints={"latency": {"max": 150}, "throughput": {"min": 8000}}
    )
    upir2.add_evidence(Evidence("benchmark", "test", 
                               {"latency": 75, "throughput": 12000}, confidence=0.85))
    architectures.append(upir2)
    
    # Streaming Pattern Variant 3: Kinesis-based
    upir3 = UPIR(id="stream_3", name="Kinesis Streaming Pipeline")
    upir3.architecture = Architecture(
        components=[
            {"name": "kinesis", "type": "queue", "config": {"shards": 8, "retention": 24}},
            {"name": "lambda", "type": "processor", "config": {"parallelism": 25, "window_size": 45}},
            {"name": "elasticache", "type": "cache", "config": {"size_mb": 768, "ttl": 450}},
            {"name": "dynamodb", "type": "storage", "config": {"replicas": 3, "sharding": True}}
        ],
        connections=[
            {"source": "kinesis", "target": "lambda"},
            {"source": "lambda", "target": "elasticache"},
            {"source": "lambda", "target": "dynamodb"},
            {"source": "elasticache", "target": "dynamodb", "condition": "cache_miss"}
        ],
        deployment={"cloud": "aws", "region": "us-west-2"},
        patterns=["streaming", "caching", "sharding", "serverless"]
    )
    upir3.specification = FormalSpecification(
        invariants=[TemporalProperty(TemporalOperator.ALWAYS, "data_consistency")],
        properties=[],
        constraints={"latency": {"max": 80}, "throughput": {"min": 12000}}
    )
    upir3.add_evidence(Evidence("production", "production",
                               {"latency": 60, "throughput": 18000}, confidence=0.95))
    architectures.append(upir3)
    
    return architectures


def create_microservices_architectures() -> list:
    """Create several microservices architectures with variations."""
    architectures = []
    
    # Microservices Pattern Variant 1: API Gateway
    upir1 = UPIR(id="micro_1", name="API Gateway Microservices")
    upir1.architecture = Architecture(
        components=[
            {"name": "api_gateway", "type": "gateway", "config": {"rate_limit": 10000, "auth": True}},
            {"name": "user_service", "type": "service", "config": {"replicas": 3, "cpu": 2}},
            {"name": "order_service", "type": "service", "config": {"replicas": 5, "cpu": 4}},
            {"name": "payment_service", "type": "service", "config": {"replicas": 2, "cpu": 2}},
            {"name": "mysql", "type": "storage", "config": {"replicas": 2, "backup": True}}
        ],
        connections=[
            {"source": "api_gateway", "target": "user_service"},
            {"source": "api_gateway", "target": "order_service"},
            {"source": "order_service", "target": "payment_service"},
            {"source": "user_service", "target": "mysql"},
            {"source": "order_service", "target": "mysql"},
            {"source": "payment_service", "target": "mysql"}
        ],
        deployment={"orchestrator": "kubernetes", "nodes": 10},
        patterns=["microservices", "api_gateway", "service_mesh"]
    )
    upir1.specification = FormalSpecification(
        invariants=[TemporalProperty(TemporalOperator.ALWAYS, "service_availability")],
        properties=[],
        constraints={"latency": {"max": 200}, "availability": {"min": 0.999}}
    )
    upir1.add_evidence(Evidence("production", "production",
                               {"latency": 150, "availability": 0.9995}, confidence=0.88))
    architectures.append(upir1)
    
    # Microservices Pattern Variant 2: Event-Driven
    upir2 = UPIR(id="micro_2", name="Event-Driven Microservices")
    upir2.architecture = Architecture(
        components=[
            {"name": "event_bus", "type": "queue", "config": {"topics": 10, "retention": 7}},
            {"name": "user_service", "type": "service", "config": {"replicas": 4, "cpu": 2}},
            {"name": "order_service", "type": "service", "config": {"replicas": 6, "cpu": 4}},
            {"name": "notification_service", "type": "service", "config": {"replicas": 2, "cpu": 1}},
            {"name": "mongodb", "type": "storage", "config": {"replicas": 3, "sharding": True}}
        ],
        connections=[
            {"source": "user_service", "target": "event_bus"},
            {"source": "order_service", "target": "event_bus"},
            {"source": "event_bus", "target": "notification_service"},
            {"source": "user_service", "target": "mongodb"},
            {"source": "order_service", "target": "mongodb"}
        ],
        deployment={"orchestrator": "docker_swarm", "nodes": 8},
        patterns=["microservices", "event_driven", "cqrs"]
    )
    upir2.specification = FormalSpecification(
        invariants=[TemporalProperty(TemporalOperator.ALWAYS, "service_availability")],
        properties=[],
        constraints={"latency": {"max": 250}, "availability": {"min": 0.99}}
    )
    upir2.add_evidence(Evidence("benchmark", "test",
                               {"latency": 180, "availability": 0.995}, confidence=0.82))
    architectures.append(upir2)
    
    return architectures


def create_batch_architectures() -> list:
    """Create batch processing architectures."""
    architectures = []
    
    # Batch Pattern: ETL Pipeline
    upir1 = UPIR(id="batch_1", name="ETL Batch Pipeline")
    upir1.architecture = Architecture(
        components=[
            {"name": "scheduler", "type": "orchestrator", "config": {"interval": 3600}},
            {"name": "extractor", "type": "processor", "config": {"parallelism": 10}},
            {"name": "transformer", "type": "processor", "config": {"parallelism": 20}},
            {"name": "loader", "type": "processor", "config": {"parallelism": 5}},
            {"name": "data_warehouse", "type": "storage", "config": {"replicas": 2}}
        ],
        connections=[
            {"source": "scheduler", "target": "extractor"},
            {"source": "extractor", "target": "transformer"},
            {"source": "transformer", "target": "loader"},
            {"source": "loader", "target": "data_warehouse"}
        ],
        deployment={"framework": "airflow", "workers": 30},
        patterns=["batch", "etl", "data_pipeline"]
    )
    upir1.specification = FormalSpecification(
        invariants=[TemporalProperty(TemporalOperator.EVENTUALLY, "job_completed")],
        properties=[],
        constraints={"completion_time": {"max": 7200}}
    )
    upir1.add_evidence(Evidence("production", "production",
                               {"completion_time": 5400}, confidence=0.91))
    architectures.append(upir1)
    
    return architectures


def demonstrate_pattern_discovery():
    """
    Main demonstration of pattern discovery and extraction.
    """
    print("=" * 70)
    print("Pattern Discovery and Extraction Demonstration")
    print("=" * 70)
    print()
    
    # Step 1: Create diverse architectures
    print("Step 1: Creating diverse architectures...")
    
    all_architectures = []
    all_architectures.extend(create_streaming_architectures())
    all_architectures.extend(create_microservices_architectures())
    all_architectures.extend(create_batch_architectures())
    
    print(f"  Created {len(all_architectures)} architectures:")
    print(f"    - {len(create_streaming_architectures())} streaming pipelines")
    print(f"    - {len(create_microservices_architectures())} microservices systems")
    print(f"    - {len(create_batch_architectures())} batch processing systems")
    print()
    
    # Step 2: Extract features
    print("Step 2: Extracting features from architectures...")
    
    feature_extractor = FeatureExtractor()
    features_list = []
    
    for upir in all_architectures:
        features = feature_extractor.extract(upir)
        features_list.append(features)
        
        print(f"  {upir.name}:")
        print(f"    Components: {features.num_components}, Connections: {features.num_connections}")
        print(f"    Uses caching: {features.uses_caching}, Uses queuing: {features.uses_queuing}")
    print()
    
    # Step 3: Cluster architectures
    print("Step 3: Clustering similar architectures...")
    
    clusterer = PatternClusterer(method="dbscan")
    clusters = clusterer.cluster(features_list, min_cluster_size=2)
    
    print(f"  Found {len(clusters)} clusters:")
    for cluster_id, indices in clusters.items():
        cluster_names = [all_architectures[i].name for i in indices]
        print(f"    Cluster {cluster_id}: {', '.join(cluster_names)}")
    print()
    
    # Step 4: Abstract patterns
    print("Step 4: Abstracting patterns from clusters...")
    
    abstractor = PatternAbstractor()
    discovered_patterns = []
    
    for cluster_id, indices in clusters.items():
        cluster_upirs = [all_architectures[i] for i in indices]
        
        # Abstract the cluster into a pattern
        pattern = abstractor.abstract(
            cluster_upirs,
            name=f"pattern_{cluster_id}"
        )
        discovered_patterns.append(pattern)
        
        print(f"  Pattern: {pattern.name}")
        print(f"    Category: {pattern.category}")
        print(f"    Instances: {len(pattern.instances)}")
        print(f"    Components: {len(pattern.template_components)}")
        print(f"    Required properties: {pattern.required_properties}")
        print(f"    Parameters: {list(pattern.parameters.keys())}")
        print()
    
    # Step 5: Create pattern library
    print("Step 5: Building pattern library...")
    
    library = PatternLibrary(storage_path="./pattern_library")
    
    for pattern in discovered_patterns:
        library.add_pattern(pattern)
    
    stats = library.get_statistics()
    print(f"  Library statistics:")
    print(f"    Total patterns: {stats['total_patterns']}")
    print(f"    Categories: {dict(stats['categories'])}")
    print(f"    Average success rate: {stats['avg_success_rate']:.2%}")
    print()
    
    # Step 6: Pattern search
    print("Step 6: Searching for patterns...")
    
    # Search for streaming patterns
    query = SearchQuery(
        category="streaming",
        component_types=["queue", "processor", "cache"],
        min_success_rate=0.8
    )
    
    results = library.search(query)
    print(f"  Search results for streaming patterns:")
    for pattern, score in results:
        print(f"    {pattern.name}: relevance={score:.2f}, success_rate={pattern.success_rate:.2%}")
    print()
    
    # Step 7: Pattern recommendation
    print("Step 7: Pattern recommendation for new system...")
    
    # Create a new UPIR that needs a pattern
    new_upir = UPIR(name="New Streaming System")
    new_upir.specification = FormalSpecification(
        invariants=[
            TemporalProperty(TemporalOperator.ALWAYS, "data_consistency"),
            TemporalProperty(TemporalOperator.WITHIN, "low_latency", time_bound=100)
        ],
        properties=[],
        constraints={"latency": {"max": 100}, "throughput": {"min": 10000}}
    )
    
    recommendations = library.recommend(new_upir, top_k=3)
    print(f"  Recommended patterns for '{new_upir.name}':")
    for pattern, score in recommendations:
        print(f"    {pattern.name}: score={score:.2f}")
        print(f"      Category: {pattern.category}")
        print(f"      Success rate: {pattern.success_rate:.2%}")
    print()
    
    # Step 8: Pattern instantiation
    if recommendations:
        print("Step 8: Instantiating recommended pattern...")
        
        best_pattern = recommendations[0][0]
        print(f"  Using pattern: {best_pattern.name}")
        
        # Instantiate with specific parameters
        params = {
            "instance_id": "prod_001",
            "processor_parallelism": 30,
            "processor_window_size": 60,
            "cache_size_mb": 2048,
            "cache_ttl": 600
        }
        
        new_architecture = best_pattern.instantiate(params)
        print(f"  Created architecture with {len(new_architecture.components)} components")
        print(f"  Components:")
        for comp in new_architecture.components[:3]:  # Show first 3
            print(f"    - {comp.get('name', 'unnamed')}: {comp.get('type', 'unknown')}")
        print()
    
    # Step 9: Record usage and evolve
    print("Step 9: Recording usage and evolving patterns...")
    
    # Simulate usage of patterns
    for pattern in discovered_patterns[:2]:
        # Record successful usage
        library.record_usage(
            pattern_id=pattern.id,
            upir_id="test_usage_1",
            success=True,
            performance={"latency": 45, "throughput": 15000},
            feedback="Pattern worked well for high-throughput scenario"
        )
        
        # Record failed usage
        library.record_usage(
            pattern_id=pattern.id,
            upir_id="test_usage_2",
            success=False,
            performance={"latency": 200, "throughput": 5000},
            feedback="Pattern struggled with burst traffic"
        )
    
    # Evolve patterns based on usage
    evolved = library.evolve_patterns(min_usage=2)
    print(f"  Evolved {len(evolved)} patterns based on usage")
    
    # Show updated statistics
    stats = library.get_statistics()
    print(f"  Updated library statistics:")
    print(f"    Total usage recorded: {stats['total_usage']}")
    if stats['most_used']:
        print(f"    Most used pattern: {stats['most_used']}")
    if stats['most_successful']:
        print(f"    Most successful pattern: {stats['most_successful']}")
    print()
    
    # Step 10: Save library
    print("Step 10: Saving pattern library...")
    library.save()
    print("  Library saved to ./pattern_library/")
    print()
    
    print("=" * 70)
    print("Pattern Discovery Complete!")
    print()
    print("Key achievements:")
    print(f"• Discovered {len(discovered_patterns)} patterns from {len(all_architectures)} architectures")
    print(f"• Patterns cover {len(stats['categories'])} categories")
    print(f"• Patterns can be instantiated with custom parameters")
    print(f"• Library provides search and recommendation capabilities")
    print(f"• Patterns evolve based on real usage feedback")
    print("=" * 70)


def demonstrate_pattern_matching():
    """
    Demonstrate pattern matching and similarity detection.
    """
    print("\n" + "=" * 70)
    print("Pattern Matching Demonstration")
    print("=" * 70)
    print()
    
    # Create a library with some patterns
    library = PatternLibrary()
    
    # Add streaming pattern
    streaming_pattern = abstractor = PatternAbstractor()
    streaming_upirs = create_streaming_architectures()
    streaming_pattern = abstractor.abstract(streaming_upirs, "streaming_master")
    library.add_pattern(streaming_pattern)
    
    # Create a new architecture to match
    test_arch = Architecture(
        components=[
            {"name": "rabbitmq", "type": "queue", "config": {"queues": 5}},
            {"name": "processor", "type": "processor", "config": {"threads": 10}},
            {"name": "redis", "type": "cache", "config": {"size": 512}},
            {"name": "postgres", "type": "storage", "config": {"replicas": 2}}
        ],
        connections=[
            {"source": "rabbitmq", "target": "processor"},
            {"source": "processor", "target": "redis"},
            {"source": "processor", "target": "postgres"}
        ],
        deployment={},
        patterns=[]
    )
    
    print("Testing pattern matching:")
    print(f"  Architecture has {len(test_arch.components)} components")
    print(f"  Component types: {[c['type'] for c in test_arch.components]}")
    print()
    
    # Check if it matches the streaming pattern
    matches = streaming_pattern.matches(test_arch, threshold=0.7)
    print(f"  Matches streaming pattern: {matches}")
    
    if matches:
        print("  ✓ This architecture follows the streaming pattern!")
        print("    Can be optimized using proven streaming configurations")
    else:
        print("  ✗ This architecture doesn't match the streaming pattern")
    
    print("=" * 70)


if __name__ == "__main__":
    # Run main demonstration
    demonstrate_pattern_discovery()
    
    # Run pattern matching demonstration
    demonstrate_pattern_matching()