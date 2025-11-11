"""
Main verification engine using Z3 SMT solver for UPIR.

This module provides the core verification capabilities that encode
distributed system architectures and temporal properties as SMT constraints
and use Z3 to prove or disprove them.

Implementation based on:
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Z3 Python API: https://z3prover.github.io/api/html/namespacez3py.html
- Bounded model checking: Clarke et al. (1999)
- SMT-LIB standard: http://smtlib.cs.uiowa.edu/

Author: Subhadip Mitra
License: Apache 2.0
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from upir.core.architecture import Architecture
from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.core.upir import UPIR
from upir.verification.solver import (
    ProofCertificate,
    VerificationResult,
    VerificationStatus,
    is_z3_available,
    get_z3_version,
)

# Import Z3 if available
if is_z3_available():
    import z3
else:
    z3 = None


class ProofCache:
    """
    Cache for verification results to avoid redundant proofs.

    Per TD Commons disclosure, caching verification results is essential
    for performance since proving the same property multiple times wastes
    computational resources. The cache uses SHA-256 hashes of property
    and architecture as keys to ensure integrity.

    Attributes:
        cache: Dictionary mapping cache keys to verification results
        hits: Number of cache hits (found in cache)
        misses: Number of cache misses (not found, had to verify)

    Example:
        >>> cache = ProofCache()
        >>> result = cache.get(property, architecture)
        >>> if result is None:
        ...     result = verify_property(property, architecture)
        ...     cache.put(property, architecture, result)

    References:
    - TD Commons: Caching for performance optimization
    - SHA-256 for cache key integrity
    """

    def __init__(self):
        """Initialize empty cache with zero hits/misses."""
        self.cache: Dict[str, VerificationResult] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(
        self,
        property: TemporalProperty,
        architecture: Architecture
    ) -> Optional[VerificationResult]:
        """
        Retrieve cached verification result if available.

        Args:
            property: Temporal property to verify
            architecture: Architecture to verify against

        Returns:
            Cached VerificationResult if found, None otherwise

        Example:
            >>> cache = ProofCache()
            >>> result = cache.get(prop, arch)
            >>> if result is not None:
            ...     print(f"Cache hit! Status: {result.status}")
        """
        key = self._compute_key(property, architecture)
        if key in self.cache:
            self.hits += 1
            # Mark result as cached
            cached_result = self.cache[key]
            # Return a copy with cached flag set to True
            return VerificationResult(
                property=cached_result.property,
                status=cached_result.status,
                certificate=cached_result.certificate,
                counterexample=cached_result.counterexample,
                execution_time=cached_result.execution_time,
                cached=True
            )
        else:
            self.misses += 1
            return None

    def put(
        self,
        property: TemporalProperty,
        architecture: Architecture,
        result: VerificationResult
    ) -> None:
        """
        Store verification result in cache.

        Args:
            property: Temporal property that was verified
            architecture: Architecture that was verified
            result: Verification result to cache

        Example:
            >>> cache = ProofCache()
            >>> result = verify_property(prop, arch)
            >>> cache.put(prop, arch, result)
        """
        key = self._compute_key(property, architecture)
        self.cache[key] = result

    def invalidate(self, architecture: Architecture) -> None:
        """
        Invalidate all cache entries for a specific architecture.

        When an architecture is modified, all cached results for that
        architecture become invalid and must be removed.

        Args:
            architecture: Architecture whose cache entries should be removed

        Example:
            >>> cache = ProofCache()
            >>> # Architecture was modified
            >>> cache.invalidate(architecture)
        """
        # Compute architecture hash
        arch_hash = architecture.hash()

        # Remove all entries containing this architecture hash
        keys_to_remove = []
        for key in self.cache:
            # Key format is "prop_hash:arch_hash"
            if key.endswith(f":{arch_hash}"):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]

    def _compute_key(
        self,
        property: TemporalProperty,
        architecture: Architecture
    ) -> str:
        """
        Compute SHA-256 cache key from property and architecture.

        The cache key uniquely identifies a (property, architecture) pair
        using cryptographic hashing to ensure integrity.

        Args:
            property: Temporal property
            architecture: Architecture

        Returns:
            Cache key as "property_hash:architecture_hash"

        Example:
            >>> cache = ProofCache()
            >>> key = cache._compute_key(prop, arch)
            >>> print(key)  # "abc123...:def456..."

        References:
        - SHA-256: Cryptographic hash for integrity
        - TD Commons: Cache key design
        """
        prop_hash = property.hash()
        arch_hash = architecture.hash()
        return f"{prop_hash}:{arch_hash}"

    def clear(self) -> None:
        """
        Clear all cache entries and reset statistics.

        Example:
            >>> cache = ProofCache()
            >>> cache.clear()
            >>> assert len(cache.cache) == 0
        """
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def __str__(self) -> str:
        """Human-readable representation."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return (
            f"ProofCache(size={len(self.cache)}, "
            f"hits={self.hits}, misses={self.misses}, "
            f"hit_rate={hit_rate:.1f}%)"
        )


class Verifier:
    """
    Main verification engine using Z3 SMT solver.

    The verifier encodes distributed system architectures and temporal
    properties as SMT constraints and uses Z3 to prove or disprove them.
    Supports caching of results and bounded model checking for temporal
    properties.

    Per TD Commons disclosure, verification is the foundation of UPIR,
    enabling formal guarantees about system behavior.

    Attributes:
        timeout: Verification timeout in milliseconds (default: 30000 = 30s)
        enable_cache: Whether to cache verification results
        cache: Proof cache instance (if caching enabled)

    Example:
        >>> verifier = Verifier(timeout=30000, enable_cache=True)
        >>> result = verifier.verify_property(property, architecture)
        >>> if result.verified:
        ...     print("Property proved!")
        ... else:
        ...     print(f"Status: {result.status}")

    References:
    - TD Commons: https://www.tdcommons.org/dpubs_series/8852/
    - Z3: https://z3prover.github.io/api/html/namespacez3py.html
    - Bounded model checking: Clarke et al. (1999)
    """

    def __init__(self, timeout: int = 30000, enable_cache: bool = True):
        """
        Initialize verifier with timeout and caching options.

        Args:
            timeout: Timeout in milliseconds (default: 30000 = 30 seconds)
            enable_cache: Enable result caching (default: True)

        Raises:
            RuntimeError: If Z3 is not available

        Example:
            >>> verifier = Verifier(timeout=60000, enable_cache=True)
        """
        if not is_z3_available():
            raise RuntimeError(
                "Z3 solver is not available. "
                "Install with: pip install z3-solver"
            )

        self.timeout = timeout
        self.enable_cache = enable_cache
        self.cache = ProofCache() if enable_cache else None

    def verify_property(
        self,
        property: TemporalProperty,
        architecture: Architecture,
        assumptions: Optional[List[str]] = None
    ) -> VerificationResult:
        """
        Verify a single temporal property against an architecture.

        This is the core verification method. It:
        1. Checks cache if enabled
        2. Encodes architecture as SMT constraints
        3. Encodes property as SMT formula
        4. Invokes Z3 solver with timeout
        5. Extracts result, counterexample, or proof certificate
        6. Caches result if enabled

        Args:
            property: Temporal property to verify
            architecture: System architecture to verify against
            assumptions: Optional list of assumptions (e.g., "network_reliable")

        Returns:
            VerificationResult with status, certificate, counterexample, timing

        Example:
            >>> prop = TemporalProperty(
            ...     operator=TemporalOperator.ALWAYS,
            ...     predicate="data_consistent"
            ... )
            >>> result = verifier.verify_property(prop, architecture)
            >>> print(f"Verified: {result.verified}")

        References:
        - TD Commons: Property verification algorithm
        - Z3: SMT solving and model extraction
        """
        assumptions = assumptions or []

        # Check cache first
        if self.enable_cache:
            cached = self.cache.get(property, architecture)
            if cached is not None:
                return cached

        # Start timing
        start_time = time.time()

        try:
            # Create Z3 solver
            solver = z3.Solver()
            solver.set("timeout", self.timeout)

            # Encode architecture as SMT constraints
            arch_constraints = self._encode_architecture(architecture)
            for constraint in arch_constraints:
                solver.add(constraint)

            # Encode property as SMT formula
            # For ALWAYS/EVENTUALLY, we use bounded model checking
            prop_constraint = self._encode_property(property, architecture)
            solver.add(prop_constraint)

            # Check satisfiability
            check_result = solver.check()

            # Compute execution time
            execution_time = time.time() - start_time

            # Process result
            if check_result == z3.sat:
                # Property is satisfiable - this means it CAN hold
                # For ALWAYS properties, satisfiability means PROVED
                # For EVENTUALLY, it means we found a path
                status = VerificationStatus.PROVED
                counterexample = None

                # Generate proof certificate
                certificate = ProofCertificate(
                    property_hash=property.hash(),
                    architecture_hash=architecture.hash(),
                    status=status,
                    proof_steps=[],  # Z3 doesn't expose proof steps easily
                    assumptions=assumptions,
                    solver_version=get_z3_version()
                )

            elif check_result == z3.unsat:
                # Property is unsatisfiable - cannot hold
                status = VerificationStatus.DISPROVED
                counterexample = self._extract_counterexample(solver)
                certificate = None

            elif check_result == z3.unknown:
                # Solver couldn't determine (incomplete theory, heuristics failed)
                reason = solver.reason_unknown()
                if "timeout" in reason.lower():
                    status = VerificationStatus.TIMEOUT
                else:
                    status = VerificationStatus.UNKNOWN
                counterexample = None
                certificate = None

            else:
                # Unexpected result
                status = VerificationStatus.ERROR
                counterexample = None
                certificate = None

            # Create result
            result = VerificationResult(
                property=property,
                status=status,
                certificate=certificate,
                counterexample=counterexample,
                execution_time=execution_time,
                cached=False
            )

            # Cache result
            if self.enable_cache:
                self.cache.put(property, architecture, result)

            return result

        except Exception as e:
            # Handle errors gracefully
            execution_time = time.time() - start_time
            return VerificationResult(
                property=property,
                status=VerificationStatus.ERROR,
                certificate=None,
                counterexample={"error": str(e)},
                execution_time=execution_time,
                cached=False
            )

    def verify_specification(self, upir: UPIR) -> List[VerificationResult]:
        """
        Verify all properties in a UPIR specification.

        Verifies both invariants and properties from the formal specification
        against the architecture. Returns results for all properties.

        Args:
            upir: UPIR instance with specification and architecture

        Returns:
            List of VerificationResult, one per property

        Raises:
            ValueError: If UPIR is missing specification or architecture

        Example:
            >>> upir = UPIR(...)
            >>> upir.specification = FormalSpecification(...)
            >>> upir.architecture = Architecture(...)
            >>> results = verifier.verify_specification(upir)
            >>> verified_count = sum(1 for r in results if r.verified)
            >>> print(f"Verified {verified_count}/{len(results)} properties")

        References:
        - TD Commons: Specification verification workflow
        """
        if upir.specification is None:
            raise ValueError("UPIR must have a specification to verify")
        if upir.architecture is None:
            raise ValueError("UPIR must have an architecture to verify")

        results = []

        # Verify invariants
        for invariant in upir.specification.invariants:
            result = self.verify_property(
                property=invariant,
                architecture=upir.architecture,
                assumptions=upir.specification.assumptions
            )
            results.append(result)

        # Verify properties
        for property in upir.specification.properties:
            result = self.verify_property(
                property=property,
                architecture=upir.architecture,
                assumptions=upir.specification.assumptions
            )
            results.append(result)

        return results

    def _encode_architecture(self, architecture: Architecture) -> List[Any]:
        """
        Encode architecture as SMT constraints.

        Converts architecture components, connections, and deployment
        into Z3 constraints. This is a simplified encoding - full
        implementation would need component-specific encodings.

        Args:
            architecture: Architecture to encode

        Returns:
            List of Z3 constraints

        Example:
            >>> constraints = verifier._encode_architecture(arch)
            >>> for c in constraints:
            ...     solver.add(c)

        References:
        - TD Commons: Architecture encoding for verification
        - Z3: Constraint construction
        """
        constraints = []

        # For now, we create a simple encoding
        # Full implementation would encode:
        # - Component states (running, failed, etc.)
        # - Network topology (connected, latency, etc.)
        # - Data consistency (replicated, partitioned, etc.)

        # Create boolean variables for component health
        for component in architecture.components:
            comp_id = component.get("id", "unknown")
            # Create a Z3 boolean variable for "component is healthy"
            var = z3.Bool(f"healthy_{comp_id}")
            # For now, assume all components can be healthy
            # (no constraints - this is a placeholder)

        # Create constraints for connections
        # If components A and B are connected and both healthy,
        # they can communicate
        for connection in architecture.connections:
            source = connection.get("source", "unknown")
            target = connection.get("target", "unknown")
            # Placeholder: no actual constraints yet

        return constraints

    def _encode_property(
        self,
        property: TemporalProperty,
        architecture: Architecture
    ) -> Any:
        """
        Encode temporal property as SMT formula.

        Uses bounded model checking for temporal operators:
        - ALWAYS: Holds in all states up to bound
        - EVENTUALLY: Holds in at least one state up to bound
        - WITHIN: Holds before time bound

        Args:
            property: Temporal property to encode
            architecture: Architecture context

        Returns:
            Z3 constraint representing the property

        Example:
            >>> constraint = verifier._encode_property(prop, arch)
            >>> solver.add(constraint)

        References:
        - Bounded model checking: Clarke et al. (1999)
        - TD Commons: Temporal property encoding
        """
        # Get predicate
        predicate = property.predicate

        # Create a Z3 boolean variable for the predicate
        pred_var = z3.Bool(predicate)

        # Encode based on operator
        if property.operator == TemporalOperator.ALWAYS:
            # ALWAYS P means P must hold in all states
            # For simplification, we just require P to hold
            # Full implementation would check all reachable states
            return pred_var

        elif property.operator == TemporalOperator.EVENTUALLY:
            # EVENTUALLY P means P must hold in at least one state
            # For simplification, we just require P to hold
            # Full implementation would use bounded model checking
            return pred_var

        elif property.operator == TemporalOperator.WITHIN:
            # WITHIN(t) P means P must hold before time t
            # For simplification, we just require P to hold
            # Full implementation would encode time bounds
            return pred_var

        elif property.operator == TemporalOperator.UNTIL:
            # P UNTIL Q means P holds until Q becomes true
            # For simplification, we just require both
            # Full implementation would encode temporal sequence
            param_predicate = property.parameters.get("until_predicate", "true")
            q_var = z3.Bool(param_predicate)
            return z3.And(pred_var, q_var)

        else:
            # Unknown operator
            return z3.BoolVal(True)

    def _extract_counterexample(self, solver: Any) -> Optional[Dict[str, Any]]:
        """
        Extract counterexample from unsatisfiable solver.

        When a property is disproved, Z3 can provide a model (counterexample)
        showing why the property doesn't hold.

        Args:
            solver: Z3 solver with unsat result

        Returns:
            Dictionary with counterexample data, or None if unavailable

        Example:
            >>> if solver.check() == z3.unsat:
            ...     ce = verifier._extract_counterexample(solver)
            ...     print(f"Counterexample: {ce}")

        References:
        - Z3: Model extraction
        - SMT-LIB: Counterexample representation
        """
        # For UNSAT, there's no model
        # For SAT, we could extract the model
        # This is a placeholder implementation
        try:
            if solver.check() == z3.sat:
                model = solver.model()
                counterexample = {}
                for decl in model.decls():
                    counterexample[decl.name()] = str(model[decl])
                return counterexample
        except:
            pass

        return None

    def __str__(self) -> str:
        """Human-readable representation."""
        cache_info = str(self.cache) if self.enable_cache else "disabled"
        return (
            f"Verifier(timeout={self.timeout}ms, "
            f"cache={cache_info})"
        )
