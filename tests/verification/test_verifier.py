"""
Unit tests for main verification engine using Z3.

Tests verify:
- ProofCache: get, put, invalidate, key computation, statistics
- Verifier: property verification, specification verification, caching
- SMT encoding: architecture and property encoding
- Error handling: timeouts, errors, missing Z3

Author: Subhadip Mitra
License: Apache 2.0
"""

import pytest
from upir.core.architecture import Architecture
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.core.upir import UPIR
from upir.verification.solver import VerificationStatus, is_z3_available
from upir.verification.verifier import ProofCache, Verifier


class TestProofCache:
    """Tests for ProofCache class."""

    def test_create_cache(self):
        """Test creating a proof cache."""
        cache = ProofCache()
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_miss(self):
        """Test cache miss on first access."""
        cache = ProofCache()
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        arch = Architecture(components=[{"id": "c1"}])

        result = cache.get(prop, arch)
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_cache_put_and_get(self):
        """Test putting and getting from cache."""
        cache = ProofCache()
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        arch = Architecture(components=[{"id": "c1"}])

        # Create a mock result
        from upir.verification.solver import VerificationResult
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED,
            execution_time=1.0
        )

        # Put in cache
        cache.put(prop, arch, result)
        assert len(cache.cache) == 1

        # Get from cache
        cached = cache.get(prop, arch)
        assert cached is not None
        assert cached.status == VerificationStatus.PROVED
        assert cached.cached is True  # Should be marked as cached
        assert cache.hits == 1

    def test_cache_key_deterministic(self):
        """Test that cache key is deterministic."""
        cache = ProofCache()
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        arch = Architecture(components=[{"id": "c1"}])

        key1 = cache._compute_key(prop, arch)
        key2 = cache._compute_key(prop, arch)
        assert key1 == key2

    def test_cache_key_different_properties(self):
        """Test that different properties have different keys."""
        cache = ProofCache()
        prop1 = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test1"
        )
        prop2 = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test2"
        )
        arch = Architecture(components=[{"id": "c1"}])

        key1 = cache._compute_key(prop1, arch)
        key2 = cache._compute_key(prop2, arch)
        assert key1 != key2

    def test_cache_key_different_architectures(self):
        """Test that different architectures have different keys."""
        cache = ProofCache()
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        arch1 = Architecture(components=[{"id": "c1"}])
        arch2 = Architecture(components=[{"id": "c2"}])

        key1 = cache._compute_key(prop, arch1)
        key2 = cache._compute_key(prop, arch2)
        assert key1 != key2

    def test_cache_invalidate(self):
        """Test invalidating cache entries for an architecture."""
        cache = ProofCache()
        arch = Architecture(components=[{"id": "c1"}])
        prop1 = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test1"
        )
        prop2 = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test2"
        )

        from upir.verification.solver import VerificationResult
        result = VerificationResult(
            property=prop1,
            status=VerificationStatus.PROVED
        )

        # Put two results for same architecture
        cache.put(prop1, arch, result)
        cache.put(prop2, arch, result)
        assert len(cache.cache) == 2

        # Invalidate architecture
        cache.invalidate(arch)
        assert len(cache.cache) == 0

    def test_cache_invalidate_selective(self):
        """Test that invalidation only affects specified architecture."""
        cache = ProofCache()
        arch1 = Architecture(components=[{"id": "c1"}])
        arch2 = Architecture(components=[{"id": "c2"}])
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )

        from upir.verification.solver import VerificationResult
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED
        )

        # Put results for two architectures
        cache.put(prop, arch1, result)
        cache.put(prop, arch2, result)
        assert len(cache.cache) == 2

        # Invalidate only arch1
        cache.invalidate(arch1)
        assert len(cache.cache) == 1

        # Verify arch2 still in cache
        cached = cache.get(prop, arch2)
        assert cached is not None

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = ProofCache()
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        arch = Architecture(components=[{"id": "c1"}])

        from upir.verification.solver import VerificationResult
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED
        )

        cache.put(prop, arch, result)
        cache.get(prop, arch)  # Generate a hit
        assert cache.hits > 0
        assert len(cache.cache) > 0

        cache.clear()
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_str(self):
        """Test string representation."""
        cache = ProofCache()
        s = str(cache)
        assert "ProofCache" in s
        assert "size=0" in s
        assert "hits=0" in s
        assert "misses=0" in s

    def test_cache_hit_rate(self):
        """Test hit rate calculation in string representation."""
        cache = ProofCache()
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        arch = Architecture(components=[{"id": "c1"}])

        # Generate some hits and misses
        cache.get(prop, arch)  # miss
        from upir.verification.solver import VerificationResult
        result = VerificationResult(
            property=prop,
            status=VerificationStatus.PROVED
        )
        cache.put(prop, arch, result)
        cache.get(prop, arch)  # hit
        cache.get(prop, arch)  # hit

        s = str(cache)
        # 2 hits, 1 miss = 66.7% hit rate
        assert "hit_rate=" in s


@pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
class TestVerifier:
    """Tests for Verifier class (requires Z3)."""

    def test_create_verifier(self):
        """Test creating a verifier."""
        verifier = Verifier(timeout=30000, enable_cache=True)
        assert verifier.timeout == 30000
        assert verifier.enable_cache is True
        assert verifier.cache is not None

    def test_create_verifier_no_cache(self):
        """Test creating verifier without cache."""
        verifier = Verifier(timeout=10000, enable_cache=False)
        assert verifier.timeout == 10000
        assert verifier.enable_cache is False
        assert verifier.cache is None

    def test_verify_simple_property(self):
        """Test verifying a simple property."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistent"
        )
        arch = Architecture(components=[{"id": "db1"}])

        result = verifier.verify_property(prop, arch)
        assert result is not None
        assert isinstance(result.status, VerificationStatus)
        assert result.execution_time > 0
        assert result.cached is False

    def test_verify_with_assumptions(self):
        """Test verifying with assumptions."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="available"
        )
        arch = Architecture(components=[{"id": "server"}])
        assumptions = ["network_reliable", "no_failures"]

        result = verifier.verify_property(prop, arch, assumptions)
        assert result is not None
        # Certificate should include assumptions if proved
        if result.certificate is not None:
            assert result.certificate.assumptions == assumptions

    def test_verify_always_operator(self):
        """Test verifying ALWAYS property."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="consistent"
        )
        arch = Architecture(components=[{"id": "c1"}])

        result = verifier.verify_property(prop, arch)
        assert result.property.operator == TemporalOperator.ALWAYS

    def test_verify_eventually_operator(self):
        """Test verifying EVENTUALLY property."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="completed"
        )
        arch = Architecture(components=[{"id": "c1"}])

        result = verifier.verify_property(prop, arch)
        assert result.property.operator == TemporalOperator.EVENTUALLY

    def test_verify_within_operator(self):
        """Test verifying WITHIN property."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="backup_done",
            time_bound=3600.0
        )
        arch = Architecture(components=[{"id": "backup_server"}])

        result = verifier.verify_property(prop, arch)
        assert result.property.operator == TemporalOperator.WITHIN
        assert result.property.time_bound == 3600.0

    def test_verify_until_operator(self):
        """Test verifying UNTIL property."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.UNTIL,
            predicate="processing",
            parameters={"until_predicate": "complete"}
        )
        arch = Architecture(components=[{"id": "processor"}])

        result = verifier.verify_property(prop, arch)
        assert result.property.operator == TemporalOperator.UNTIL

    def test_verify_with_cache_miss(self):
        """Test first verification is a cache miss."""
        verifier = Verifier(timeout=5000, enable_cache=True)
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        arch = Architecture(components=[{"id": "c1"}])

        assert verifier.cache.misses == 0
        result = verifier.verify_property(prop, arch)
        assert result.cached is False
        assert verifier.cache.misses == 1

    def test_verify_with_cache_hit(self):
        """Test second verification is a cache hit."""
        verifier = Verifier(timeout=5000, enable_cache=True)
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        arch = Architecture(components=[{"id": "c1"}])

        # First verification (miss)
        result1 = verifier.verify_property(prop, arch)
        assert result1.cached is False

        # Second verification (hit)
        result2 = verifier.verify_property(prop, arch)
        assert result2.cached is True
        assert verifier.cache.hits == 1

    def test_verify_cached_result_same_status(self):
        """Test cached result has same status as original."""
        verifier = Verifier(timeout=5000, enable_cache=True)
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        arch = Architecture(components=[{"id": "c1"}])

        result1 = verifier.verify_property(prop, arch)
        result2 = verifier.verify_property(prop, arch)
        assert result1.status == result2.status

    def test_verify_specification_success(self):
        """Test verifying a complete specification."""
        verifier = Verifier(timeout=5000, enable_cache=False)

        # Create UPIR with specification and architecture
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="data_valid"
                )
            ],
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.EVENTUALLY,
                    predicate="backup_complete"
                )
            ],
            assumptions=["network_reliable"]
        )
        arch = Architecture(components=[{"id": "server"}])
        upir = UPIR(
            id="test-upir",
            name="Test UPIR",
            description="Test",
            specification=spec,
            architecture=arch
        )

        results = verifier.verify_specification(upir)
        assert len(results) == 2  # 1 invariant + 1 property
        assert all(isinstance(r.status, VerificationStatus) for r in results)

    def test_verify_specification_missing_spec(self):
        """Test error when UPIR has no specification."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        upir = UPIR(
            id="test",
            name="Test",
            description="Test"
        )

        with pytest.raises(ValueError, match="must have a specification"):
            verifier.verify_specification(upir)

    def test_verify_specification_missing_architecture(self):
        """Test error when UPIR has no architecture."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        spec = FormalSpecification()
        upir = UPIR(
            id="test",
            name="Test",
            description="Test",
            specification=spec
        )

        with pytest.raises(ValueError, match="must have an architecture"):
            verifier.verify_specification(upir)

    def test_encode_architecture(self):
        """Test architecture encoding."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        arch = Architecture(
            components=[
                {"id": "web", "type": "frontend"},
                {"id": "db", "type": "database"}
            ],
            connections=[
                {"source": "web", "target": "db"}
            ]
        )

        constraints = verifier._encode_architecture(arch)
        assert isinstance(constraints, list)
        # Currently returns empty list (placeholder)
        # Full implementation would have actual constraints

    def test_encode_always_property(self):
        """Test encoding ALWAYS property."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="consistent"
        )
        arch = Architecture(components=[{"id": "c1"}])

        constraint = verifier._encode_property(prop, arch)
        assert constraint is not None
        # Z3 constraint object

    def test_encode_eventually_property(self):
        """Test encoding EVENTUALLY property."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="complete"
        )
        arch = Architecture(components=[{"id": "c1"}])

        constraint = verifier._encode_property(prop, arch)
        assert constraint is not None

    def test_encode_within_property(self):
        """Test encoding WITHIN property."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="done",
            time_bound=60.0
        )
        arch = Architecture(components=[{"id": "c1"}])

        constraint = verifier._encode_property(prop, arch)
        assert constraint is not None

    def test_encode_until_property(self):
        """Test encoding UNTIL property."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.UNTIL,
            predicate="running",
            parameters={"until_predicate": "stopped"}
        )
        arch = Architecture(components=[{"id": "c1"}])

        constraint = verifier._encode_property(prop, arch)
        assert constraint is not None

    def test_verifier_str(self):
        """Test string representation."""
        verifier = Verifier(timeout=15000, enable_cache=True)
        s = str(verifier)
        assert "Verifier" in s
        assert "15000ms" in s
        assert "cache=" in s

    def test_verifier_str_no_cache(self):
        """Test string representation without cache."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        s = str(verifier)
        assert "Verifier" in s
        assert "cache=disabled" in s


class TestVerifierWithoutZ3:
    """Tests for Verifier behavior when Z3 is not available."""

    def test_verifier_requires_z3(self):
        """Test that Verifier raises error if Z3 not available."""
        if is_z3_available():
            pytest.skip("Z3 is available, cannot test unavailable case")

        with pytest.raises(RuntimeError, match="Z3 solver is not available"):
            Verifier()


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
    def test_verify_empty_architecture(self):
        """Test verifying against empty architecture."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        arch = Architecture()  # Empty

        result = verifier.verify_property(prop, arch)
        assert result is not None

    @pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
    def test_verify_complex_architecture(self):
        """Test verifying against complex architecture."""
        verifier = Verifier(timeout=5000, enable_cache=False)
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="available"
        )
        arch = Architecture(
            components=[
                {"id": f"server{i}", "type": "compute"}
                for i in range(10)
            ],
            connections=[
                {"source": f"server{i}", "target": f"server{i+1}"}
                for i in range(9)
            ],
            deployment={"region": "us-west", "replicas": 3}
        )

        result = verifier.verify_property(prop, arch)
        assert result is not None

    @pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
    def test_cache_with_multiple_properties(self):
        """Test cache with multiple different properties."""
        verifier = Verifier(timeout=5000, enable_cache=True)
        arch = Architecture(components=[{"id": "c1"}])

        props = [
            TemporalProperty(
                operator=TemporalOperator.ALWAYS,
                predicate=f"prop{i}"
            )
            for i in range(5)
        ]

        # Verify all properties
        for prop in props:
            verifier.verify_property(prop, arch)

        # Cache should have 5 entries
        assert len(verifier.cache.cache) == 5
        assert verifier.cache.misses == 5

        # Verify again - should all be hits
        for prop in props:
            result = verifier.verify_property(prop, arch)
            assert result.cached is True

        assert verifier.cache.hits == 5

    def test_cache_key_format(self):
        """Test cache key format is property_hash:architecture_hash."""
        cache = ProofCache()
        prop = TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="test"
        )
        arch = Architecture(components=[{"id": "c1"}])

        key = cache._compute_key(prop, arch)
        assert ":" in key
        parts = key.split(":")
        assert len(parts) == 2
        assert len(parts[0]) == 64  # SHA-256 hash length
        assert len(parts[1]) == 64  # SHA-256 hash length
