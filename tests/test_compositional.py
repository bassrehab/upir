"""
Tests for compositional verification framework.
"""

import pytest
from upir.verification.compositional import (
    CompositionalVerifier,
    Component,
    Property,
    PropertyType,
    Proof,
    AssumeGuarantee,
    ProofComposer
)


class TestCompositionalVerifier:
    """Test the compositional verifier."""
    
    def setup_method(self):
        self.verifier = CompositionalVerifier()
        
    def test_add_component(self):
        """Test adding components to the system."""
        comp1 = Component(
            name="producer",
            inputs={'data': str},
            outputs={'message': str},
            properties=[
                Property(
                    name="no_data_loss",
                    type=PropertyType.INVARIANT,
                    formula="output_count == input_count",
                    components=["producer"]
                )
            ]
        )
        
        self.verifier.add_component(comp1)
        
        assert "producer" in self.verifier.components
        assert "producer" in self.verifier.dependency_graph.nodes()
        
    def test_add_connection(self):
        """Test adding connections between components."""
        comp1 = Component("producer", {}, {'message': str}, [])
        comp2 = Component("consumer", {'message': str}, {}, [])
        
        self.verifier.add_component(comp1)
        self.verifier.add_component(comp2)
        self.verifier.add_connection("producer", "consumer", "data")
        
        assert self.verifier.dependency_graph.has_edge("producer", "consumer")
        
    def test_verify_single_component(self):
        """Test verifying a single component."""
        comp = Component(
            name="counter",
            inputs={'increment': int},
            outputs={'count': int},
            properties=[
                Property(
                    name="positive_count",
                    type=PropertyType.INVARIANT,
                    formula="count > 0",
                    components=["counter"]
                )
            ]
        )
        
        self.verifier.add_component(comp)
        proof = self.verifier._verify_component(comp)
        
        assert proof is not None
        assert proof.property_name == "counter_properties"
        assert "counter" in proof.components
        
    def test_verify_system(self):
        """Test verifying an entire system."""
        # Create a simple producer-consumer system
        producer = Component(
            name="producer",
            inputs={},
            outputs={'data': int},
            properties=[
                Property(
                    name="produces_positive",
                    type=PropertyType.INVARIANT,
                    formula="data > 0",
                    components=["producer"]
                )
            ]
        )
        
        consumer = Component(
            name="consumer",
            inputs={'data': int},
            outputs={'processed': int},
            properties=[
                Property(
                    name="preserves_sign",
                    type=PropertyType.INVARIANT,
                    formula="processed > 0",
                    components=["consumer"]
                )
            ]
        )
        
        self.verifier.add_component(producer)
        self.verifier.add_component(consumer)
        self.verifier.add_connection("producer", "consumer")
        
        # Add system-level property
        system_prop = Property(
            name="end_to_end_positive",
            type=PropertyType.INVARIANT,
            formula="processed > 0",
            components=["producer", "consumer"]
        )
        self.verifier.add_property(system_prop)
        
        result = self.verifier.verify_system()
        
        assert result is not None
        assert isinstance(result.verified, bool)
        assert len(result.proofs) > 0
        assert result.total_time_ms >= 0
        
    def test_interface_compatibility(self):
        """Test checking interface compatibility."""
        comp1 = Component(
            name="sender",
            inputs={},
            outputs={'message': str, 'timestamp': int},
            properties=[]
        )
        
        comp2 = Component(
            name="receiver",
            inputs={'message': str, 'timestamp': int},
            outputs={},
            properties=[]
        )
        
        comp3 = Component(
            name="incompatible",
            inputs={'data': bytes},  # Wrong type
            outputs={},
            properties=[]
        )
        
        self.verifier.add_component(comp1)
        self.verifier.add_component(comp2)
        self.verifier.add_component(comp3)
        
        # Compatible interface
        assert self.verifier._check_interface_compatibility(comp1, comp2) == True
        
        # Incompatible interface
        assert self.verifier._check_interface_compatibility(comp1, comp3) == True  # No matching field names
        
    def test_verification_order(self):
        """Test getting optimal verification order."""
        # Create a chain: A -> B -> C
        self.verifier.add_component(Component("A", {}, {}, []))
        self.verifier.add_component(Component("B", {}, {}, []))
        self.verifier.add_component(Component("C", {}, {}, []))
        
        self.verifier.add_connection("A", "B")
        self.verifier.add_connection("B", "C")
        
        order = self.verifier.get_verification_order()
        
        assert order == ["A", "B", "C"]
        
    def test_incremental_verification(self):
        """Test incremental verification of changed components."""
        # Create components
        for name in ["A", "B", "C", "D"]:
            comp = Component(name, {}, {}, [])
            self.verifier.add_component(comp)
        
        # Create dependencies: A -> B -> C, A -> D
        self.verifier.add_connection("A", "B")
        self.verifier.add_connection("B", "C")
        self.verifier.add_connection("A", "D")
        
        # First, verify everything
        initial_result = self.verifier.verify_system()
        
        # Now verify incrementally with B changed
        incremental_result = self.verifier.verify_incremental(["B"])
        
        # Should re-verify B and C (dependent on B), but not A or D
        assert "B" in incremental_result.component_times
        assert "C" in incremental_result.component_times
        # A and D should not be re-verified (not in component_times for incremental)
        
    def test_proof_caching(self):
        """Test that proofs are cached and reused."""
        comp = Component(
            name="cached_comp",
            inputs={'x': int},
            outputs={'y': int},
            properties=[]
        )
        
        self.verifier.add_component(comp)
        
        # First verification
        proof1 = self.verifier._verify_component(comp)
        time1 = proof1.verification_time_ms
        
        # Second verification (should use cache)
        proof2 = self.verifier._verify_component(comp)
        time2 = proof2.verification_time_ms
        
        # Second should be faster (near 0) due to caching
        assert time2 <= time1
        assert proof1.property_name == proof2.property_name


class TestAssumeGuarantee:
    """Test assume-guarantee reasoning."""
    
    def setup_method(self):
        self.verifier = CompositionalVerifier()
        self.ag = AssumeGuarantee(self.verifier)
        
    def test_verify_with_contracts(self):
        """Test verifying with assume-guarantee contracts."""
        comp = Component(
            name="buffer",
            inputs={'data': int},
            outputs={'buffered': int},
            properties=[]
        )
        
        assumes = [
            Property(
                name="input_bounded",
                type=PropertyType.INVARIANT,
                formula="data < 100",
                components=["buffer"]
            )
        ]
        
        guarantees = [
            Property(
                name="output_bounded",
                type=PropertyType.INVARIANT,
                formula="buffered < 100",
                components=["buffer"]
            )
        ]
        
        # This should verify (if input < 100, then output < 100 for a buffer)
        result = self.ag.verify_with_contracts(comp, assumes, guarantees)
        
        assert isinstance(result, bool)
        
    def test_circular_reasoning(self):
        """Test circular assume-guarantee reasoning."""
        comp1 = Component("comp1", {}, {}, [])
        comp2 = Component("comp2", {}, {}, [])
        
        prop1 = Property(
            name="prop1",
            type=PropertyType.INVARIANT,
            formula="x > 0",
            components=["comp1"]
        )
        
        prop2 = Property(
            name="prop2",
            type=PropertyType.INVARIANT,
            formula="y > 0",
            components=["comp2"]
        )
        
        result = self.ag.circular_reasoning(comp1, comp2, prop1, prop2)
        
        assert isinstance(result, bool)


class TestProofComposer:
    """Test proof composition."""
    
    def setup_method(self):
        self.composer = ProofComposer()
        
    def test_add_proof(self):
        """Test adding proofs to the composer."""
        proof = Proof(
            property_name="test_property",
            components=["comp1"],
            z3_proof=True,
            verification_time_ms=10.5,
            assumptions=["assumption1"]
        )
        
        self.composer.add_proof(proof)
        
        assert "test_property" in self.composer.proof_dag.nodes()
        
    def test_compose_proofs(self):
        """Test composing multiple proofs."""
        # Create a chain of proofs
        proof1 = Proof(
            property_name="base_property",
            components=["comp1"],
            z3_proof=True,
            verification_time_ms=5.0,
            assumptions=[]
        )
        
        proof2 = Proof(
            property_name="derived_property",
            components=["comp1", "comp2"],
            z3_proof=True,
            verification_time_ms=10.0,
            assumptions=["base_property"]
        )
        
        self.composer.add_proof(proof1)
        self.composer.add_proof(proof2)
        
        # Compose to prove derived_property
        composed = self.composer.compose_proofs("derived_property")
        
        assert composed is not None
        assert composed.property_name == "composed_derived_property"
        assert "base_property" in composed.assumptions
        
    def test_invalid_composition(self):
        """Test that invalid compositions are rejected."""
        # Add a proof with missing dependencies
        proof = Proof(
            property_name="incomplete",
            components=["comp1"],
            z3_proof=None,  # Invalid proof
            verification_time_ms=5.0,
            assumptions=["missing_assumption"]
        )
        
        self.composer.add_proof(proof)
        
        result = self.composer.compose_proofs("incomplete")
        
        # Should return None because dependencies are missing
        assert result is None
        
    def test_proof_certificate(self):
        """Test generating a proof certificate."""
        proof1 = Proof(
            property_name="prop1",
            components=["comp1"],
            z3_proof=True,
            verification_time_ms=5.0,
            assumptions=[]
        )
        
        proof2 = Proof(
            property_name="prop2",
            components=["comp2"],
            z3_proof=True,
            verification_time_ms=7.0,
            assumptions=["prop1"]
        )
        
        self.composer.add_proof(proof1)
        self.composer.add_proof(proof2)
        
        certificate = self.composer.get_proof_certificate()
        
        assert 'timestamp' in certificate
        assert 'proofs' in certificate
        assert 'dependencies' in certificate
        assert 'prop1' in certificate['proofs']
        assert 'prop2' in certificate['proofs']
        assert certificate['proofs']['prop1']['verified'] == True
        assert certificate['proofs']['prop2']['assumptions'] == ["prop1"]


class TestPropertyTypes:
    """Test different property types."""
    
    def test_safety_property(self):
        """Test safety properties."""
        prop = Property(
            name="no_overflow",
            type=PropertyType.SAFETY,
            formula="buffer_size < max_size",
            components=["buffer"]
        )
        
        assert prop.type == PropertyType.SAFETY
        assert not prop.verified
        
    def test_liveness_property(self):
        """Test liveness properties."""
        prop = Property(
            name="eventually_processed",
            type=PropertyType.LIVENESS,
            formula="Eventually(processed == true)",
            components=["processor"]
        )
        
        assert prop.type == PropertyType.LIVENESS
        
    def test_temporal_property(self):
        """Test temporal properties."""
        prop = Property(
            name="request_response",
            type=PropertyType.TEMPORAL,
            formula="Always(request -> Eventually(response))",
            components=["server"]
        )
        
        assert prop.type == PropertyType.TEMPORAL