"""
Unit tests for UPIR core models.

Testing the data structures that form the foundation of UPIR.
Focus on serialization, validation, and core operations.

Author: subhadipmitra@google.com
"""

import json
import pytest
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upir.core.models import (
    TemporalOperator, TemporalProperty, FormalSpecification,
    Evidence, ReasoningNode, SynthesisProof, Implementation,
    Architecture, UPIR, ConfidenceLevel
)


class TestTemporalProperty:
    """Test temporal property functionality."""
    
    def test_temporal_property_creation(self):
        """Test creating temporal properties."""
        prop = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="all_events_processed",
            time_bound=100.0,
            parameters={"threshold": 0.95}
        )
        
        assert prop.operator == TemporalOperator.EVENTUALLY
        assert prop.predicate == "all_events_processed"
        assert prop.time_bound == 100.0
        assert prop.parameters["threshold"] == 0.95
    
    def test_temporal_property_to_smt(self):
        """Test SMT formula generation."""
        # Test ALWAYS operator
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        smt = prop.to_smt()
        assert "forall" in smt
        assert "data_consistent" in smt
        
        # Test EVENTUALLY with time bound
        prop = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="processed",
            time_bound=50.0
        )
        smt = prop.to_smt()
        assert "exists" in smt
        assert "<= t 50" in smt
    
    def test_temporal_property_serialization(self):
        """Test serialization and deserialization."""
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="latency_bound",
            time_bound=100.0,
            parameters={"max_latency": 100}
        )
        
        # Serialize
        data = prop.to_dict()
        assert data["operator"] == "within"
        assert data["time_bound"] == 100.0
        
        # Deserialize
        prop2 = TemporalProperty.from_dict(data)
        assert prop2.operator == prop.operator
        assert prop2.predicate == prop.predicate
        assert prop2.time_bound == prop.time_bound


class TestFormalSpecification:
    """Test formal specification functionality."""
    
    def test_specification_creation(self):
        """Test creating a formal specification."""
        invariants = [
            TemporalProperty(TemporalOperator.ALWAYS, "consistent"),
            TemporalProperty(TemporalOperator.WITHIN, "processed", 100.0)
        ]
        
        constraints = {
            "latency": {"max": 100},
            "cost": {"max": 5000}
        }
        
        spec = FormalSpecification(
            invariants=invariants,
            properties=[],
            constraints=constraints,
            assumptions=["network_reliable"]
        )
        
        assert len(spec.invariants) == 2
        assert spec.constraints["latency"]["max"] == 100
        assert "network_reliable" in spec.assumptions
    
    def test_specification_validation(self):
        """Test specification validation."""
        # Valid specification
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(TemporalOperator.ALWAYS, "prop1"),
                TemporalProperty(TemporalOperator.ALWAYS, "prop2")
            ],
            properties=[],
            constraints={"latency": {"max": 100}}
        )
        assert spec.validate() is True
        
        # Invalid: duplicate invariants
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(TemporalOperator.ALWAYS, "prop1"),
                TemporalProperty(TemporalOperator.ALWAYS, "prop1")  # Duplicate
            ],
            properties=[],
            constraints={}
        )
        assert spec.validate() is False
        
        # Invalid: bad constraint format
        spec = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={"latency": {"invalid_key": 100}}
        )
        assert spec.validate() is False


class TestEvidence:
    """Test evidence and Bayesian confidence updates."""
    
    def test_evidence_creation(self):
        """Test creating evidence."""
        evidence = Evidence(
            source="benchmark_test",
            type="benchmark",
            data={"latency": 50, "throughput": 10000},
            confidence=0.8
        )
        
        assert evidence.source == "benchmark_test"
        assert evidence.confidence == 0.8
        assert evidence.data["latency"] == 50
    
    def test_bayesian_confidence_update(self):
        """Test Bayesian confidence updates."""
        evidence = Evidence(
            source="test",
            type="test",
            data={},
            confidence=0.5
        )
        
        # Positive observation should increase confidence
        initial_confidence = evidence.confidence
        evidence.update_confidence(True, prior_weight=0.1)
        assert evidence.confidence > initial_confidence
        assert evidence.confidence <= 1.0
        
        # Negative observation should decrease confidence
        initial_confidence = evidence.confidence
        evidence.update_confidence(False, prior_weight=0.1)
        assert evidence.confidence < initial_confidence
        assert evidence.confidence >= 0.0
    
    def test_confidence_bounds(self):
        """Test that confidence stays within [0, 1]."""
        # Test upper bound
        evidence = Evidence("test", "test", {}, confidence=0.95)
        for _ in range(100):  # Many positive observations
            evidence.update_confidence(True, prior_weight=0.1)
        assert evidence.confidence <= 1.0
        
        # Test lower bound
        evidence = Evidence("test", "test", {}, confidence=0.05)
        for _ in range(100):  # Many negative observations
            evidence.update_confidence(False, prior_weight=0.1)
        assert evidence.confidence >= 0.0


class TestReasoningNode:
    """Test reasoning DAG nodes."""
    
    def test_reasoning_node_creation(self):
        """Test creating reasoning nodes."""
        node = ReasoningNode(
            decision="Use streaming architecture",
            rationale="Low latency requirements",
            evidence_ids=["ev1", "ev2"],
            parent_ids=["parent1"],
            alternatives=[{"option": "batch", "reason": "too slow"}],
            confidence=0.8
        )
        
        assert node.decision == "Use streaming architecture"
        assert len(node.evidence_ids) == 2
        assert len(node.parent_ids) == 1
        assert node.confidence == 0.8
    
    def test_confidence_computation(self):
        """Test computing confidence from evidence."""
        # Create evidence
        evidence_map = {
            "ev1": Evidence("source1", "test", {}, confidence=0.8),
            "ev2": Evidence("source2", "test", {}, confidence=0.9),
            "ev3": Evidence("source3", "test", {}, confidence=0.7)
        }
        
        node = ReasoningNode(
            decision="Decision",
            rationale="Rationale",
            evidence_ids=["ev1", "ev2", "ev3"]
        )
        
        # Compute confidence (should be geometric mean)
        confidence = node.compute_confidence(evidence_map)
        
        # Geometric mean of 0.8, 0.9, 0.7 ≈ 0.793
        assert 0.79 < confidence < 0.80


class TestUPIR:
    """Test the main UPIR class."""
    
    def test_upir_creation(self):
        """Test creating a UPIR instance."""
        upir = UPIR(
            name="Test System",
            description="A test distributed system"
        )
        
        assert upir.name == "Test System"
        assert upir.description == "A test distributed system"
        assert upir.id is not None  # UUID should be generated
    
    def test_add_evidence(self):
        """Test adding evidence to UPIR."""
        upir = UPIR()
        evidence = Evidence("test", "test", {"data": 1}, 0.5)
        
        eid = upir.add_evidence(evidence)
        
        assert eid in upir.evidence
        assert upir.evidence[eid] == evidence
    
    def test_add_reasoning(self):
        """Test adding reasoning nodes to UPIR."""
        upir = UPIR()
        node = ReasoningNode(
            decision="Test decision",
            rationale="Test rationale"
        )
        
        nid = upir.add_reasoning(node)
        
        assert nid in upir.reasoning
        assert upir.reasoning[nid] == node
    
    def test_dag_validation(self):
        """Test DAG cycle detection."""
        upir = UPIR()
        
        # Create nodes with cycle
        node1 = ReasoningNode(id="n1", decision="D1", rationale="R1", parent_ids=["n3"])
        node2 = ReasoningNode(id="n2", decision="D2", rationale="R2", parent_ids=["n1"])
        node3 = ReasoningNode(id="n3", decision="D3", rationale="R3", parent_ids=["n2"])
        
        upir.reasoning["n1"] = node1
        upir.reasoning["n2"] = node2
        upir.reasoning["n3"] = node3
        
        # Should detect cycle
        assert upir._validate_dag() is False
        
        # Create valid DAG
        upir2 = UPIR()
        node1 = ReasoningNode(id="n1", decision="D1", rationale="R1", parent_ids=[])
        node2 = ReasoningNode(id="n2", decision="D2", rationale="R2", parent_ids=["n1"])
        node3 = ReasoningNode(id="n3", decision="D3", rationale="R3", parent_ids=["n2"])
        
        upir2.reasoning["n1"] = node1
        upir2.reasoning["n2"] = node2
        upir2.reasoning["n3"] = node3
        
        # Should be valid
        assert upir2._validate_dag() is True
    
    def test_overall_confidence_computation(self):
        """Test computing overall confidence."""
        upir = UPIR()
        
        # Add evidence
        ev1 = Evidence("s1", "test", {}, confidence=0.8)
        ev2 = Evidence("s2", "test", {}, confidence=0.9)
        
        eid1 = upir.add_evidence(ev1)
        eid2 = upir.add_evidence(ev2)
        
        # Add reasoning nodes (leaf nodes)
        node1 = ReasoningNode(
            decision="D1",
            rationale="R1",
            evidence_ids=[eid1]
        )
        node2 = ReasoningNode(
            decision="D2",
            rationale="R2",
            evidence_ids=[eid2]
        )
        
        upir.add_reasoning(node1)
        upir.add_reasoning(node2)
        
        # Compute overall confidence
        confidence = upir.compute_overall_confidence()
        
        # Should be harmonic mean of leaf node confidences
        assert confidence > 0
        assert confidence <= 1
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        # Create a complete UPIR
        upir = UPIR(name="Test", description="Test system")
        
        # Add specification
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(TemporalOperator.ALWAYS, "consistent")
            ],
            properties=[],
            constraints={"latency": {"max": 100}}
        )
        upir.specification = spec
        
        # Add architecture
        arch = Architecture(
            components=[{"name": "comp1"}],
            connections=[],
            deployment={"env": "prod"},
            patterns=["streaming"]
        )
        upir.architecture = arch
        
        # Add evidence
        evidence = Evidence("test", "test", {"data": 1}, 0.8)
        upir.add_evidence(evidence)
        
        # Serialize to JSON
        json_str = upir.to_json()
        data = json.loads(json_str)
        
        # Check key fields
        assert data["name"] == "Test"
        assert data["specification"] is not None
        assert data["architecture"] is not None
        assert len(data["evidence"]) == 1
        assert "signature" in data
        
        # Deserialize
        upir2 = UPIR.from_json(json_str)
        
        assert upir2.name == upir.name
        assert upir2.specification.invariants[0].predicate == "consistent"
        assert len(upir2.evidence) == 1
    
    def test_signature_generation(self):
        """Test cryptographic signature generation."""
        upir1 = UPIR(name="System1")
        upir2 = UPIR(name="System2")
        
        sig1 = upir1.generate_signature()
        sig2 = upir2.generate_signature()
        
        # Different UPIRs should have different signatures
        assert sig1 != sig2
        
        # Same UPIR should have consistent signature
        sig1_again = upir1.generate_signature()
        assert sig1 == sig1_again


class TestSynthesisProof:
    """Test synthesis proof functionality."""
    
    def test_synthesis_proof_creation(self):
        """Test creating a synthesis proof."""
        proof = SynthesisProof(
            specification_hash="spec_hash",
            implementation_hash="impl_hash",
            proof_steps=[
                {"step": 1, "action": "Generate skeleton"},
                {"step": 2, "action": "Fill holes"}
            ],
            verification_result=True
        )
        
        assert proof.specification_hash == "spec_hash"
        assert proof.verification_result is True
        assert len(proof.proof_steps) == 2
    
    def test_certificate_generation(self):
        """Test generating cryptographic certificate."""
        proof = SynthesisProof(
            specification_hash="spec123",
            implementation_hash="impl456",
            proof_steps=[],
            verification_result=True
        )
        
        cert = proof.generate_certificate()
        
        # Should be a valid hash
        assert len(cert) == 64  # SHA-256 produces 64 hex characters
        
        # Same proof should generate same certificate
        cert2 = proof.generate_certificate()
        assert cert == cert2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])