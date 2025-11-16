"""
Test all code examples from documentation.

This ensures documentation stays accurate and examples work correctly.

Based on:
- docs/getting-started/quickstart.md
- docs/api/ reference pages
- docs/guide/ user guides

Author: Subhadip Mitra
License: Apache 2.0
"""

import pytest
from upir import (
    UPIR,
    Architecture,
    FormalSpecification,
    TemporalProperty,
    TemporalOperator,
    Verifier,
)
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.verification.solver import VerificationStatus
from upir.verification.verifier import Verifier
from upir.synthesis.cegis import Synthesizer
from upir.learning.learner import ArchitectureLearner
from upir.patterns.library import PatternLibrary
from upir.patterns.extractor import PatternExtractor


class TestQuickStartExamples:
    """Test examples from docs/getting-started/quickstart.md"""

    def test_quickstart_full_example(self):
        """Test the complete quick start example."""
        # Step 1: Define temporal properties
        properties = [
            # Data must always be consistent
            TemporalProperty(
                operator=TemporalOperator.ALWAYS, predicate="data_consistent"
            ),
            # Respond within 100ms
            TemporalProperty(
                operator=TemporalOperator.WITHIN, predicate="respond", time_bound=100
            ),
        ]

        # Step 2: Create formal specification
        spec = FormalSpecification(
            properties=properties,
            constraints={
                "latency_p99": {"max": 100.0},  # 100ms max
                "monthly_cost": {"max": 1000.0},  # $1000/month max
            },
        )

        # Step 3: Define architecture components
        components = [
            {
                "id": "api_gateway",
                "name": "API Gateway",
                "type": "api_gateway",
                "latency_ms": 10.0,
                "cost_monthly": 300.0,
            },
            {
                "id": "database",
                "name": "Database",
                "type": "database",
                "latency_ms": 50.0,
                "cost_monthly": 500.0,
            },
        ]

        connections = [{"from": "api_gateway", "to": "database", "latency_ms": 5.0}]

        arch = Architecture(components=components, connections=connections)

        # Step 4: Create UPIR instance
        upir = UPIR(specification=spec, architecture=arch)

        # Step 5: Verify the architecture
        verifier = Verifier()
        results = verifier.verify_specification(upir)

        # Step 6: Check results
        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert result.status in [
                VerificationStatus.PROVED,
                VerificationStatus.DISPROVED,
                VerificationStatus.UNKNOWN,
            ]

    def test_temporal_operators(self):
        """Test temporal operator examples from quickstart."""
        # ALWAYS operator
        always_consistent = TemporalProperty(
            operator=TemporalOperator.ALWAYS, predicate="data_consistent"
        )
        assert always_consistent.operator == TemporalOperator.ALWAYS

        # WITHIN operator
        low_latency = TemporalProperty(
            operator=TemporalOperator.WITHIN, predicate="respond", time_bound=100
        )
        assert low_latency.time_bound == 100

        # EVENTUALLY operator
        eventually_complete = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="all_tasks_complete",
            time_bound=60000,
        )
        assert eventually_complete.time_bound == 60000


class TestAPIReferenceExamples:
    """Test examples from docs/api/ reference pages."""

    def test_upir_api_example(self):
        """Test UPIR API example from docs/api/core/upir.md"""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS, predicate="data_consistent"
                )
            ],
            constraints={"latency_p99": {"max": 100.0}},
        )

        arch = Architecture(
            components=[{"id": "api", "type": "api_gateway", "latency_ms": 10}],
            connections=[],
        )

        upir = UPIR(specification=spec, architecture=arch)

        # Validate
        is_valid = upir.validate()
        assert isinstance(is_valid, bool)

        # Serialize
        upir_json = upir.to_json()
        assert isinstance(upir_json, str)

    def test_architecture_api_example(self):
        """Test Architecture API example from docs/api/core/architecture.md"""
        components = [
            {
                "id": "api_gateway",
                "name": "API Gateway",
                "type": "api_gateway",
                "latency_ms": 10.0,
                "cost_monthly": 300.0,
                "config": {"max_connections": 10000},
            },
            {
                "id": "database",
                "name": "PostgreSQL Database",
                "type": "database",
                "latency_ms": 50.0,
                "cost_monthly": 500.0,
                "config": {"instance_type": "db.m5.large"},
            },
        ]

        connections = [{"from": "api_gateway", "to": "database", "latency_ms": 5.0}]

        arch = Architecture(components=components, connections=connections)

        # Access metrics
        assert arch.total_latency_ms >= 0
        assert arch.total_cost >= 0
        assert len(arch.components) == 2

        # Serialize
        arch_json = arch.to_json()
        assert isinstance(arch_json, str)

    def test_specification_api_example(self):
        """Test FormalSpecification API example from docs/api/core/specification.md"""
        properties = [
            TemporalProperty(
                operator=TemporalOperator.EVENTUALLY,
                predicate="task_complete",
                time_bound=60000,
            ),
            TemporalProperty(
                operator=TemporalOperator.WITHIN,
                predicate="respond",
                time_bound=100,
            ),
        ]

        invariants = [
            TemporalProperty(
                operator=TemporalOperator.ALWAYS, predicate="data_consistent"
            ),
            TemporalProperty(
                operator=TemporalOperator.ALWAYS, predicate="no_data_loss"
            ),
        ]

        constraints = {
            "latency_p99": {"max": 100.0},
            "monthly_cost": {"max": 5000.0},
            "throughput_qps": {"min": 10000.0},
        }

        spec = FormalSpecification(
            properties=properties, invariants=invariants, constraints=constraints
        )

        # Serialize
        spec_json = spec.to_json()
        assert isinstance(spec_json, str)

    def test_temporal_api_examples(self):
        """Test temporal logic examples from docs/api/core/temporal.md"""
        # ALWAYS
        always_consistent = TemporalProperty(
            operator=TemporalOperator.ALWAYS, predicate="data_consistent"
        )
        smt = always_consistent.to_smt()
        assert isinstance(smt, str)

        # EVENTUALLY
        eventually_complete = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="all_tasks_complete",
            time_bound=60000,
        )
        assert eventually_complete.time_bound == 60000

        # WITHIN
        within_100ms = TemporalProperty(
            operator=TemporalOperator.WITHIN, predicate="respond", time_bound=100
        )
        assert within_100ms.time_bound == 100

        # UNTIL
        until_complete = TemporalProperty(
            operator=TemporalOperator.UNTIL, predicate="processing", time_bound=30000
        )
        assert until_complete.time_bound == 30000

    def test_verifier_api_example(self):
        """Test Verifier API example from docs/api/verification/verifier.md"""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS, predicate="data_consistent"
                )
            ],
            constraints={"latency_p99": {"max": 100.0}},
        )

        arch = Architecture(
            components=[{"id": "api", "type": "api_gateway", "latency_ms": 10}],
            connections=[],
        )

        upir = UPIR(specification=spec, architecture=arch)

        # Create verifier
        verifier = Verifier(timeout=10000)

        # Verify specification
        results = verifier.verify_specification(upir)

        # Check results
        assert isinstance(results, list)
        for result in results:
            assert result.status in [
                VerificationStatus.PROVED,
                VerificationStatus.DISPROVED,
                VerificationStatus.UNKNOWN,
            ]

    def test_pattern_library_api_example(self):
        """Test PatternLibrary API example from docs/api/patterns/library.md"""
        library = PatternLibrary()

        # Library should have built-in patterns
        assert len(library.patterns) > 0

        # Search by name
        results = library.search_patterns("streaming")
        assert isinstance(results, list)


class TestGuideExamples:
    """Test examples from docs/guide/ user guides."""

    def test_specifications_guide_examples(self):
        """Test examples from docs/guide/specifications.md"""
        # Safety property
        safety = TemporalProperty(
            operator=TemporalOperator.ALWAYS, predicate="data_consistent"
        )

        # Liveness property
        liveness = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="all_tasks_complete",
            time_bound=60000,
        )

        # Performance property
        performance = TemporalProperty(
            operator=TemporalOperator.WITHIN, predicate="respond", time_bound=100
        )

        # Create specification
        spec = FormalSpecification(
            properties=[safety, liveness, performance],
            constraints={
                "latency_p99": {"max": 100.0},
                "latency_p50": {"max": 50.0},
                "monthly_cost": {"max": 5000.0},
                "throughput_qps": {"min": 10000.0},
                "availability": {"min": 0.999},
            },
        )

        assert len(spec.properties) == 3
        assert len(spec.constraints) == 5

    def test_high_availability_pattern(self):
        """Test high-availability pattern from specifications guide."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS, predicate="system_available"
                )
            ],
            constraints={
                "availability": {"min": 0.9999},
                "failover_time_ms": {"max": 1000},
            },
        )

        assert len(spec.constraints) == 2
        assert spec.constraints["availability"]["min"] == 0.9999

    def test_low_latency_api_pattern(self):
        """Test low-latency API pattern from specifications guide."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.WITHIN,
                    predicate="respond",
                    time_bound=100,
                )
            ],
            constraints={
                "latency_p99": {"max": 100.0},
                "latency_p50": {"max": 50.0},
            },
        )

        assert len(spec.properties) == 1
        assert spec.properties[0].time_bound == 100


class TestExampleDocumentation:
    """Test examples from docs/examples/ documentation."""

    def test_streaming_example_spec(self):
        """Test streaming example specification from docs/examples/streaming.md"""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.EVENTUALLY,
                    predicate="all_events_processed",
                    time_bound=100000,
                ),
                TemporalProperty(
                    operator=TemporalOperator.WITHIN,
                    predicate="process_event",
                    time_bound=100,
                ),
            ],
            invariants=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS, predicate="data_consistent"
                )
            ],
            constraints={
                "latency_p99": {"max": 100.0},
                "monthly_cost": {"max": 5000.0},
                "throughput_qps": {"min": 10000.0},
            },
        )

        assert len(spec.properties) == 2
        assert len(spec.invariants) == 1
        assert len(spec.constraints) == 3

    def test_streaming_example_architecture(self):
        """Test streaming example architecture from docs/examples/streaming.md"""
        components = [
            {
                "id": "pubsub_source",
                "type": "pubsub_source",
                "latency_ms": 5.0,
                "cost_monthly": 500.0,
            },
            {
                "id": "beam_processor",
                "type": "streaming_processor",
                "latency_ms": 50.0,
                "cost_monthly": 3000.0,
            },
            {
                "id": "bigquery_sink",
                "type": "database",
                "latency_ms": 30.0,
                "cost_monthly": 1200.0,
            },
        ]

        connections = [
            {"from": "pubsub_source", "to": "beam_processor", "latency_ms": 2.0},
            {"from": "beam_processor", "to": "bigquery_sink", "latency_ms": 3.0},
        ]

        arch = Architecture(components=components, connections=connections)

        assert len(arch.components) == 3
        assert len(arch.connections) == 2
        assert arch.total_cost == 4700.0  # 500 + 3000 + 1200
