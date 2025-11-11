"""
Unit tests for main UPIR class and Architecture.

Tests verify:
- Architecture creation and serialization
- UPIR creation and validation
- Evidence and reasoning node management
- Confidence computation (harmonic mean)
- DAG cycle detection
- Signature generation
- JSON serialization/deserialization

Author: Subhadip Mitra
License: Apache 2.0
"""

from datetime import datetime

import pytest
from upir.core.architecture import Architecture
from upir.core.evidence import Evidence, ReasoningNode
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.core.upir import UPIR


class TestArchitecture:
    """Tests for Architecture dataclass."""

    def test_create_empty_architecture(self):
        """Test creating an empty architecture."""
        arch = Architecture()
        assert arch.components == []
        assert arch.connections == []
        assert arch.deployment == {}
        assert arch.patterns == []

    def test_create_architecture_with_components(self):
        """Test creating architecture with components."""
        arch = Architecture(
            components=[
                {"name": "api-service", "type": "service"},
                {"name": "postgres", "type": "database"}
            ]
        )
        assert len(arch.components) == 2
        assert arch.components[0]["name"] == "api-service"

    def test_create_architecture_complete(self):
        """Test creating complete architecture."""
        arch = Architecture(
            components=[
                {"name": "service", "replicas": 3}
            ],
            connections=[
                {"from": "service", "to": "db", "protocol": "TCP"}
            ],
            deployment={
                "regions": ["us-west-2"],
                "strategy": "rolling"
            },
            patterns=["microservices", "CQRS"]
        )
        assert len(arch.components) == 1
        assert len(arch.connections) == 1
        assert arch.deployment["strategy"] == "rolling"
        assert "CQRS" in arch.patterns

    def test_architecture_to_dict(self):
        """Test architecture serialization."""
        arch = Architecture(
            components=[{"name": "service"}],
            patterns=["microservices"]
        )
        d = arch.to_dict()
        assert d["components"][0]["name"] == "service"
        assert d["patterns"] == ["microservices"]

    def test_architecture_from_dict(self):
        """Test architecture deserialization."""
        data = {
            "components": [{"name": "service"}],
            "connections": [],
            "deployment": {"region": "us-west-2"},
            "patterns": ["microservices"]
        }
        arch = Architecture.from_dict(data)
        assert len(arch.components) == 1
        assert arch.deployment["region"] == "us-west-2"

    def test_architecture_roundtrip(self):
        """Test serialize->deserialize preserves architecture."""
        original = Architecture(
            components=[{"name": "svc", "type": "api"}],
            connections=[{"from": "a", "to": "b"}],
            deployment={"env": "prod"},
            patterns=["event-driven"]
        )
        data = original.to_dict()
        restored = Architecture.from_dict(data)

        assert restored.components == original.components
        assert restored.connections == original.connections
        assert restored.deployment == original.deployment
        assert restored.patterns == original.patterns

    def test_architecture_str(self):
        """Test __str__ representation."""
        arch = Architecture(
            components=[{"name": "s1"}, {"name": "s2"}],
            connections=[{"from": "s1", "to": "s2"}],
            patterns=["microservices"]
        )
        s = str(arch)
        assert "2 component(s)" in s
        assert "1 connection(s)" in s
        assert "1 pattern(s)" in s


class TestUPIRCreation:
    """Tests for UPIR creation and basic operations."""

    def test_create_minimal_upir(self):
        """Test creating UPIR with minimal fields."""
        upir = UPIR(
            id="upir-1",
            name="Test System",
            description="A test system"
        )
        assert upir.id == "upir-1"
        assert upir.name == "Test System"
        assert upir.specification is None
        assert upir.architecture is None
        assert upir.evidence == {}
        assert upir.reasoning == {}
        assert isinstance(upir.created_at, datetime)
        assert isinstance(upir.updated_at, datetime)

    def test_create_complete_upir(self):
        """Test creating UPIR with all fields."""
        spec = FormalSpecification()
        arch = Architecture(components=[{"name": "service"}])

        upir = UPIR(
            id="upir-1",
            name="Test",
            description="Description",
            specification=spec,
            architecture=arch,
            metadata={"owner": "team-a"}
        )
        assert upir.specification is not None
        assert upir.architecture is not None
        assert upir.metadata["owner"] == "team-a"

    def test_generate_id(self):
        """Test ID generation."""
        upir_id = UPIR.generate_id()
        assert isinstance(upir_id, str)
        assert len(upir_id) == 36  # UUID format

    def test_empty_id_rejected(self):
        """Test that empty ID is rejected."""
        with pytest.raises(ValueError, match="ID cannot be empty"):
            UPIR(id="", name="test", description="test")

    def test_empty_name_rejected(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValueError, match="Name cannot be empty"):
            UPIR(id="upir-1", name="", description="test")


class TestUPIREvidenceManagement:
    """Tests for evidence management."""

    def test_add_evidence(self):
        """Test adding evidence to UPIR."""
        upir = UPIR(id="upir-1", name="test", description="test")
        evidence = Evidence("source", "benchmark", {"metric": 100}, 0.8)

        evidence_id = upir.add_evidence(evidence)

        assert isinstance(evidence_id, str)
        assert len(evidence_id) == 36  # UUID
        assert evidence_id in upir.evidence
        assert upir.evidence[evidence_id] == evidence

    def test_add_multiple_evidence(self):
        """Test adding multiple pieces of evidence."""
        upir = UPIR(id="upir-1", name="test", description="test")

        e1 = Evidence("src1", "benchmark", {}, 0.8)
        e2 = Evidence("src2", "test", {}, 0.9)

        id1 = upir.add_evidence(e1)
        id2 = upir.add_evidence(e2)

        assert id1 != id2
        assert len(upir.evidence) == 2

    def test_add_evidence_updates_timestamp(self):
        """Test that adding evidence updates updated_at."""
        upir = UPIR(id="upir-1", name="test", description="test")
        original_time = upir.updated_at

        # Small sleep to ensure time difference
        import time
        time.sleep(0.01)

        evidence = Evidence("src", "benchmark", {}, 0.8)
        upir.add_evidence(evidence)

        assert upir.updated_at > original_time


class TestUPIRReasoningManagement:
    """Tests for reasoning node management."""

    def test_add_reasoning(self):
        """Test adding reasoning node to UPIR."""
        upir = UPIR(id="upir-1", name="test", description="test")
        node = ReasoningNode(
            id="node-1",
            decision="Use PostgreSQL",
            rationale="Strong consistency needed"
        )

        node_id = upir.add_reasoning(node)

        assert node_id == "node-1"
        assert node_id in upir.reasoning
        assert upir.reasoning[node_id] == node

    def test_add_multiple_reasoning_nodes(self):
        """Test adding multiple reasoning nodes."""
        upir = UPIR(id="upir-1", name="test", description="test")

        n1 = ReasoningNode("n1", "decision1", "rationale1")
        n2 = ReasoningNode("n2", "decision2", "rationale2")

        upir.add_reasoning(n1)
        upir.add_reasoning(n2)

        assert len(upir.reasoning) == 2


class TestUPIRConfidenceComputation:
    """Tests for overall confidence computation."""

    def test_compute_confidence_no_nodes(self):
        """Test confidence with no reasoning nodes."""
        upir = UPIR(id="upir-1", name="test", description="test")
        assert upir.compute_overall_confidence() == 0.0

    def test_compute_confidence_single_leaf(self):
        """Test confidence with single leaf node."""
        upir = UPIR(id="upir-1", name="test", description="test")
        node = ReasoningNode("n1", "decision", "rationale", confidence=0.8)
        upir.add_reasoning(node)

        conf = upir.compute_overall_confidence()
        assert abs(conf - 0.8) < 1e-10

    def test_compute_confidence_multiple_leaves(self):
        """Test confidence with multiple leaf nodes."""
        upir = UPIR(id="upir-1", name="test", description="test")

        n1 = ReasoningNode("n1", "d1", "r1", confidence=0.8)
        n2 = ReasoningNode("n2", "d2", "r2", confidence=0.9)

        upir.add_reasoning(n1)
        upir.add_reasoning(n2)

        conf = upir.compute_overall_confidence()
        # Harmonic mean of 0.8 and 0.9: 2 / (1/0.8 + 1/0.9) ≈ 0.8471
        expected = 2.0 / (1.0/0.8 + 1.0/0.9)
        assert abs(conf - expected) < 1e-10

    def test_compute_confidence_with_dag(self):
        """Test confidence with DAG (non-leaf nodes ignored)."""
        upir = UPIR(id="upir-1", name="test", description="test")

        # Create DAG: n1 <- n2 (n1 is parent of n2)
        n1 = ReasoningNode("n1", "parent decision", "r1", confidence=0.5)
        n2 = ReasoningNode(
            "n2", "leaf decision", "r2",
            parent_ids=["n1"],  # n2 depends on n1
            confidence=0.9
        )

        upir.add_reasoning(n1)
        upir.add_reasoning(n2)

        # Only n2 is a leaf (n1 is referenced as parent)
        conf = upir.compute_overall_confidence()
        assert abs(conf - 0.9) < 1e-10

    def test_compute_confidence_harmonic_mean_formula(self):
        """Test harmonic mean formula: n / sum(1/c_i)."""
        upir = UPIR(id="upir-1", name="test", description="test")

        confidences = [0.6, 0.7, 0.8, 0.9]
        for i, c in enumerate(confidences):
            node = ReasoningNode(f"n{i}", "decision", "rationale", confidence=c)
            upir.add_reasoning(node)

        conf = upir.compute_overall_confidence()

        # Manual computation
        n = len(confidences)
        reciprocal_sum = sum(1.0/c for c in confidences)
        expected = n / reciprocal_sum

        assert abs(conf - expected) < 1e-10

    def test_compute_confidence_zero_confidence(self):
        """Test that zero confidence in any leaf returns 0."""
        upir = UPIR(id="upir-1", name="test", description="test")

        n1 = ReasoningNode("n1", "d1", "r1", confidence=0.0)
        n2 = ReasoningNode("n2", "d2", "r2", confidence=0.9)

        upir.add_reasoning(n1)
        upir.add_reasoning(n2)

        assert upir.compute_overall_confidence() == 0.0


class TestUPIRValidation:
    """Tests for UPIR validation."""

    def test_validate_empty_upir(self):
        """Test validation of empty UPIR."""
        upir = UPIR(id="upir-1", name="test", description="test")
        assert upir.validate() is True

    def test_validate_with_specification(self):
        """Test validation with specification."""
        spec = FormalSpecification(
            constraints={"latency": {"max": 100}}
        )
        upir = UPIR(
            id="upir-1",
            name="test",
            description="test",
            specification=spec
        )
        assert upir.validate() is True

    def test_validate_invalid_specification(self):
        """Test that invalid specification fails validation."""
        spec = FormalSpecification(
            constraints={"latency": {"min": 100, "max": 10}}  # min > max
        )
        upir = UPIR(
            id="upir-1",
            name="test",
            description="test",
            specification=spec
        )
        with pytest.raises(ValueError, match="min .* > max"):
            upir.validate()

    def test_validate_evidence_references(self):
        """Test validation of evidence references."""
        upir = UPIR(id="upir-1", name="test", description="test")

        # Add evidence
        evidence = Evidence("src", "benchmark", {}, 0.8)
        e_id = upir.add_evidence(evidence)

        # Add node referencing evidence
        node = ReasoningNode(
            "n1", "decision", "rationale",
            evidence_ids=[e_id]
        )
        upir.add_reasoning(node)

        assert upir.validate() is True

    def test_validate_missing_evidence_reference(self):
        """Test that missing evidence reference fails validation."""
        upir = UPIR(id="upir-1", name="test", description="test")

        # Add node referencing non-existent evidence
        node = ReasoningNode(
            "n1", "decision", "rationale",
            evidence_ids=["non-existent-evidence-id"]
        )
        upir.add_reasoning(node)

        with pytest.raises(ValueError, match="references non-existent evidence"):
            upir.validate()

    def test_validate_dag_no_cycle(self):
        """Test DAG validation with no cycles."""
        upir = UPIR(id="upir-1", name="test", description="test")

        # Create simple chain: n1 <- n2 <- n3
        n1 = ReasoningNode("n1", "d1", "r1")
        n2 = ReasoningNode("n2", "d2", "r2", parent_ids=["n1"])
        n3 = ReasoningNode("n3", "d3", "r3", parent_ids=["n2"])

        upir.add_reasoning(n1)
        upir.add_reasoning(n2)
        upir.add_reasoning(n3)

        assert upir.validate() is True

    def test_validate_dag_with_cycle(self):
        """Test that cycle in DAG is detected."""
        upir = UPIR(id="upir-1", name="test", description="test")

        # Create cycle: n1 <- n2 <- n1
        n1 = ReasoningNode("n1", "d1", "r1", parent_ids=["n2"])
        n2 = ReasoningNode("n2", "d2", "r2", parent_ids=["n1"])

        upir.add_reasoning(n1)
        upir.add_reasoning(n2)

        with pytest.raises(ValueError, match="Cycle detected"):
            upir.validate()

    def test_validate_dag_self_cycle(self):
        """Test that self-referencing node is detected as cycle."""
        upir = UPIR(id="upir-1", name="test", description="test")

        # Node references itself
        n1 = ReasoningNode("n1", "d1", "r1", parent_ids=["n1"])
        upir.add_reasoning(n1)

        with pytest.raises(ValueError, match="Cycle detected"):
            upir.validate()

    def test_validate_complex_dag(self):
        """Test validation of complex DAG structure."""
        upir = UPIR(id="upir-1", name="test", description="test")

        # Diamond structure: n1 <- n2, n1 <- n3, n2 <- n4, n3 <- n4
        n1 = ReasoningNode("n1", "d1", "r1")
        n2 = ReasoningNode("n2", "d2", "r2", parent_ids=["n1"])
        n3 = ReasoningNode("n3", "d3", "r3", parent_ids=["n1"])
        n4 = ReasoningNode("n4", "d4", "r4", parent_ids=["n2", "n3"])

        upir.add_reasoning(n1)
        upir.add_reasoning(n2)
        upir.add_reasoning(n3)
        upir.add_reasoning(n4)

        assert upir.validate() is True


class TestUPIRSignature:
    """Tests for signature generation."""

    def test_generate_signature(self):
        """Test signature generation."""
        upir = UPIR(id="upir-1", name="test", description="test")
        sig = upir.generate_signature()

        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256

    def test_signature_deterministic(self):
        """Test that signature is deterministic."""
        upir = UPIR(id="upir-1", name="test", description="test")
        sig1 = upir.generate_signature()
        sig2 = upir.generate_signature()

        assert sig1 == sig2

    def test_signature_changes_with_content(self):
        """Test that signature changes when content changes."""
        upir1 = UPIR(id="upir-1", name="test", description="test1")
        upir2 = UPIR(id="upir-1", name="test", description="test2")

        assert upir1.generate_signature() != upir2.generate_signature()

    def test_signature_with_evidence(self):
        """Test signature with evidence."""
        upir = UPIR(id="upir-1", name="test", description="test")
        sig1 = upir.generate_signature()

        evidence = Evidence("src", "benchmark", {}, 0.8)
        upir.add_evidence(evidence)
        sig2 = upir.generate_signature()

        assert sig1 != sig2


class TestUPIRSerialization:
    """Tests for JSON serialization."""

    def test_to_json(self):
        """Test JSON serialization."""
        upir = UPIR(id="upir-1", name="test", description="desc")
        json_str = upir.to_json()

        assert isinstance(json_str, str)
        assert "upir-1" in json_str
        assert "test" in json_str

    def test_from_json(self):
        """Test JSON deserialization."""
        upir = UPIR(id="upir-1", name="test", description="desc")
        json_str = upir.to_json()
        restored = UPIR.from_json(json_str)

        assert restored.id == upir.id
        assert restored.name == upir.name
        assert restored.description == upir.description

    def test_roundtrip_json_minimal(self):
        """Test JSON round-trip with minimal UPIR."""
        original = UPIR(
            id="upir-1",
            name="Test System",
            description="A test system"
        )
        json_str = original.to_json()
        restored = UPIR.from_json(json_str)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.description == original.description

    def test_roundtrip_json_complete(self):
        """Test JSON round-trip with complete UPIR."""
        spec = FormalSpecification(
            constraints={"latency": {"max": 100}}
        )
        arch = Architecture(
            components=[{"name": "service"}],
            patterns=["microservices"]
        )

        original = UPIR(
            id="upir-1",
            name="Test",
            description="Desc",
            specification=spec,
            architecture=arch,
            metadata={"owner": "team-a"}
        )

        # Add evidence
        evidence = Evidence("src", "benchmark", {"metric": 100}, 0.8)
        e_id = original.add_evidence(evidence)

        # Add reasoning
        node = ReasoningNode(
            "n1", "decision", "rationale",
            evidence_ids=[e_id],
            confidence=0.9
        )
        original.add_reasoning(node)

        json_str = original.to_json()
        restored = UPIR.from_json(json_str)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.specification is not None
        assert restored.architecture is not None
        assert len(restored.evidence) == 1
        assert len(restored.reasoning) == 1
        assert restored.metadata["owner"] == "team-a"


class TestUPIRStringRepresentations:
    """Tests for string representations."""

    def test_str_minimal(self):
        """Test __str__ for minimal UPIR."""
        upir = UPIR(id="upir-1", name="Test System", description="desc")
        s = str(upir)
        assert "Test System" in s

    def test_str_complete(self):
        """Test __str__ for complete UPIR."""
        upir = UPIR(
            id="upir-1",
            name="Test",
            description="desc",
            specification=FormalSpecification(),
            architecture=Architecture()
        )
        evidence = Evidence("src", "benchmark", {}, 0.8)
        node = ReasoningNode("n1", "decision", "rationale")

        upir.add_evidence(evidence)
        upir.add_reasoning(node)

        s = str(upir)
        assert "Test" in s
        assert "with spec" in s
        assert "with arch" in s
        assert "1 evidence" in s
        assert "1 reasoning nodes" in s

    def test_repr(self):
        """Test __repr__ representation."""
        upir = UPIR(
            id="upir-1",
            name="Test",
            description="desc",
            specification=FormalSpecification()
        )
        r = repr(upir)

        assert "UPIR" in r
        assert "upir-1" in r
        assert "Test" in r
        assert "spec=True" in r
