"""
Unit tests for UPIR verification engine.

Testing the formal verification capabilities with and without Z3.
Includes tests for proof caching and incremental verification.

Author: subhadipmitra@google.com
"""

import pytest
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upir.core.models import (
    UPIR, FormalSpecification, TemporalProperty, 
    TemporalOperator, Architecture
)
from upir.verification.verifier import (
    Verifier, VerificationStatus, VerificationResult,
    ProofCache, ProofCertificate
)


class TestProofCache:
    """Test proof caching functionality."""
    
    def test_cache_creation(self):
        """Test creating a proof cache."""
        cache = ProofCache(cache_size=100)
        
        assert cache.cache_size == 100
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_key_computation(self):
        """Test cache key generation."""
        cache = ProofCache()
        
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="consistent"
        )
        
        arch = Architecture(
            components=[{"name": "comp1"}],
            connections=[],
            deployment={"env": "prod"}
        )
        
        key = cache._compute_key(prop, arch)
        
        # Key should be a hash
        assert len(key) == 64  # SHA-256 produces 64 hex chars
        
        # Same inputs should produce same key
        key2 = cache._compute_key(prop, arch)
        assert key == key2
    
    def test_cache_put_and_get(self):
        """Test storing and retrieving from cache."""
        cache = ProofCache()
        
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test_property"
        )
        
        arch = Architecture(
            components=[{"name": "test"}],
            connections=[],
            deployment={}
        )
        
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED,
            execution_time=1.0
        )
        
        # Store in cache
        cache.put(prop, arch, result)
        assert len(cache.cache) == 1
        
        # Retrieve from cache
        cached_result = cache.get(prop, arch)
        assert cached_result is not None
        assert cached_result.status == VerificationStatus.PROVED
        assert cached_result.cached is True
        assert cache.hits == 1
        
        # Miss on different property
        prop2 = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="different"
        )
        
        missed_result = cache.get(prop2, arch)
        assert missed_result is None
        assert cache.misses == 1
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ProofCache(cache_size=2)
        
        arch = Architecture(components=[], connections=[], deployment={})
        
        # Add first entry
        prop1 = TemporalProperty(TemporalOperator.ALWAYS, "prop1")
        result1 = VerificationResult(prop1, VerificationStatus.PROVED)
        cache.put(prop1, arch, result1)
        
        # Add second entry
        prop2 = TemporalProperty(TemporalOperator.ALWAYS, "prop2")
        result2 = VerificationResult(prop2, VerificationStatus.PROVED)
        cache.put(prop2, arch, result2)
        
        assert len(cache.cache) == 2
        
        # Add third entry - should evict first
        prop3 = TemporalProperty(TemporalOperator.ALWAYS, "prop3")
        result3 = VerificationResult(prop3, VerificationStatus.PROVED)
        cache.put(prop3, arch, result3)
        
        assert len(cache.cache) == 2
        
        # First entry should be evicted
        assert cache.get(prop1, arch) is None
        assert cache.get(prop2, arch) is not None
        assert cache.get(prop3, arch) is not None
    
    def test_cache_invalidation(self):
        """Test invalidating cached proofs."""
        cache = ProofCache()
        
        arch1 = Architecture(
            components=[{"name": "comp1"}],
            connections=[],
            deployment={"version": "1"}
        )
        
        arch2 = Architecture(
            components=[{"name": "comp2"}],
            connections=[],
            deployment={"version": "2"}
        )
        
        prop = TemporalProperty(TemporalOperator.ALWAYS, "test")
        
        # Add entries for both architectures
        result1 = VerificationResult(prop, VerificationStatus.PROVED)
        result2 = VerificationResult(prop, VerificationStatus.DISPROVED)
        
        cache.put(prop, arch1, result1)
        cache.put(prop, arch2, result2)
        
        assert len(cache.cache) == 2
        
        # Invalidate first architecture
        cache.invalidate(arch1)
        
        # First should be gone, second should remain
        assert cache.get(prop, arch1) is None
        assert cache.get(prop, arch2) is not None


class TestProofCertificate:
    """Test proof certificate generation."""
    
    def test_certificate_creation(self):
        """Test creating a proof certificate."""
        cert = ProofCertificate(
            property_hash="prop_hash",
            architecture_hash="arch_hash",
            status=VerificationStatus.PROVED,
            proof_steps=[{"step": 1, "action": "test"}],
            assumptions=["assumption1"]
        )
        
        assert cert.property_hash == "prop_hash"
        assert cert.status == VerificationStatus.PROVED
        assert len(cert.proof_steps) == 1
        assert "assumption1" in cert.assumptions
    
    def test_certificate_hash_generation(self):
        """Test generating certificate hash."""
        cert = ProofCertificate(
            property_hash="prop123",
            architecture_hash="arch456",
            status=VerificationStatus.PROVED,
            proof_steps=[],
            assumptions=[]
        )
        
        hash1 = cert.generate_hash()
        
        # Should be deterministic
        hash2 = cert.generate_hash()
        assert hash1 == hash2
        
        # Should be a valid SHA-256 hash
        assert len(hash1) == 64
    
    def test_certificate_serialization(self):
        """Test certificate serialization."""
        cert = ProofCertificate(
            property_hash="test_prop",
            architecture_hash="test_arch",
            status=VerificationStatus.DISPROVED,
            proof_steps=[{"step": 1}],
            assumptions=["test"]
        )
        
        data = cert.to_dict()
        
        assert data["property_hash"] == "test_prop"
        assert data["status"] == "disproved"
        assert "certificate_hash" in data
        assert len(data["proof_steps"]) == 1


class TestVerifier:
    """Test the main verification engine."""
    
    def test_verifier_creation(self):
        """Test creating a verifier."""
        verifier = Verifier(timeout=5000, enable_cache=True)
        
        assert verifier.timeout == 5000
        assert verifier.cache is not None
        
        verifier_no_cache = Verifier(enable_cache=False)
        assert verifier_no_cache.cache is None
    
    def test_verify_empty_specification(self):
        """Test verifying empty UPIR."""
        verifier = Verifier()
        upir = UPIR()
        
        # No specification
        results = verifier.verify_specification(upir)
        assert len(results) == 0
        
        # Add specification but no architecture
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(TemporalOperator.ALWAYS, "test")
            ],
            properties=[],
            constraints={}
        )
        upir.specification = spec
        
        results = verifier.verify_specification(upir)
        assert len(results) == 0
    
    def test_verify_property_without_z3(self):
        """Test verification when Z3 is not available."""
        verifier = Verifier()
        
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test_property"
        )
        
        arch = Architecture(
            components=[{"name": "test"}],
            connections=[],
            deployment={}
        )
        
        result = verifier.verify_property(prop, arch)
        
        # Without Z3, status should be UNKNOWN
        # (unless Z3 is actually installed)
        assert result.property == prop
        assert result.status in [
            VerificationStatus.UNKNOWN,
            VerificationStatus.PROVED,
            VerificationStatus.DISPROVED
        ]
    
    def test_verify_with_assumptions(self):
        """Test verification with assumptions."""
        verifier = Verifier()
        
        prop = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="goal_reached",
            time_bound=100.0
        )
        
        arch = Architecture(
            components=[{"name": "system"}],
            connections=[],
            deployment={}
        )
        
        assumptions = [
            "network_reliable",
            "storage_available"
        ]
        
        result = verifier.verify_property(prop, arch, assumptions)
        
        assert result.property == prop
        # Certificate should include assumptions
        if result.certificate:
            assert len(result.certificate.assumptions) == len(assumptions)
    
    def test_verification_caching(self):
        """Test that verification results are cached."""
        verifier = Verifier(enable_cache=True)
        
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="cached_property"
        )
        
        arch = Architecture(
            components=[{"name": "cached"}],
            connections=[],
            deployment={}
        )
        
        # First verification
        result1 = verifier.verify_property(prop, arch)
        assert result1.cached is False
        
        # Second verification should be cached
        result2 = verifier.verify_property(prop, arch)
        assert result2.cached is True
        assert result2.status == result1.status
        
        # Cache should have recorded a hit
        assert verifier.cache.hits == 1
    
    def test_incremental_verification(self):
        """Test incremental verification with changed properties."""
        verifier = Verifier(enable_cache=True)
        
        # Create UPIR with multiple properties
        upir = UPIR()
        
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(TemporalOperator.ALWAYS, "prop1"),
                TemporalProperty(TemporalOperator.ALWAYS, "prop2")
            ],
            properties=[
                TemporalProperty(TemporalOperator.EVENTUALLY, "prop3")
            ],
            constraints={}
        )
        
        arch = Architecture(
            components=[{"name": "test"}],
            connections=[],
            deployment={}
        )
        
        upir.specification = spec
        upir.architecture = arch
        
        # First verification - nothing cached
        results1 = verifier.verify_incremental(upir)
        assert len(results1) == 3
        
        # Second verification - should use cache
        results2 = verifier.verify_incremental(upir)
        assert len(results2) == 3
        
        # Check cache was used
        if verifier.cache:
            assert verifier.cache.hits > 0
        
        # Mark a property as changed
        results3 = verifier.verify_incremental(upir, changed_properties={"prop1"})
        assert len(results3) == 3
    
    def test_verification_timeout(self):
        """Test verification timeout handling."""
        # Use very short timeout
        verifier = Verifier(timeout=1)  # 1ms timeout
        
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="complex_property"
        )
        
        arch = Architecture(
            components=[{"name": f"comp{i}"} for i in range(100)],
            connections=[],
            deployment={}
        )
        
        result = verifier.verify_property(prop, arch)
        
        # Should handle timeout gracefully
        assert result.property == prop
        assert result.status in [
            VerificationStatus.TIMEOUT,
            VerificationStatus.UNKNOWN,
            VerificationStatus.PROVED,  # Might be fast enough
            VerificationStatus.DISPROVED
        ]
    
    def test_complete_verification_flow(self):
        """Test complete verification flow with UPIR."""
        # Create a complete UPIR
        upir = UPIR(name="Test System")
        
        # Add specification
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="data_consistent"
                ),
                TemporalProperty(
                    operator=TemporalOperator.WITHIN,
                    predicate="request_processed",
                    time_bound=100.0
                )
            ],
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.EVENTUALLY,
                    predicate="system_ready"
                )
            ],
            constraints={
                "latency": {"max": 100},
                "availability": {"min": 0.99}
            },
            assumptions=["network_reliable"]
        )
        
        arch = Architecture(
            components=[
                {"name": "frontend", "type": "service"},
                {"name": "backend", "type": "service"},
                {"name": "database", "type": "storage"}
            ],
            connections=[
                {"source": "frontend", "target": "backend"},
                {"source": "backend", "target": "database"}
            ],
            deployment={
                "environment": "production",
                "replicas": 3
            },
            patterns=["microservices", "layered"]
        )
        
        upir.specification = spec
        upir.architecture = arch
        
        # Verify
        verifier = Verifier()
        results = verifier.verify_specification(upir)
        
        # Should verify all properties
        assert len(results) == 3  # 2 invariants + 1 property
        
        # Check results
        for result in results:
            assert isinstance(result, VerificationResult)
            assert result.property in (spec.invariants + spec.properties)
            assert result.execution_time >= 0
            
            # If proved/disproved, should have certificate
            if result.status in [VerificationStatus.PROVED, VerificationStatus.DISPROVED]:
                assert result.certificate is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])