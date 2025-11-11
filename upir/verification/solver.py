"""
SMT-based verification using Z3 theorem prover for UPIR.

This module provides formal verification capabilities using the Z3 SMT solver
to prove or disprove temporal properties about distributed system architectures.

Implementation based on:
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Z3 GitHub: https://github.com/Z3Prover/z3
- Z3 Python API: https://z3prover.github.io/api/html/namespacez3py.html
- Python hashlib: https://docs.python.org/3/library/hashlib.html

Author: Subhadip Mitra
License: Apache 2.0
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from upir.core.temporal import TemporalProperty

# Try to import Z3, but allow graceful degradation if not available
try:
    import z3
    Z3_AVAILABLE = True
    Z3_VERSION = z3.get_version_string()
except ImportError:
    Z3_AVAILABLE = False
    Z3_VERSION = "unavailable"


class VerificationStatus(Enum):
    """
    Status of a formal verification attempt.

    Based on standard SMT solver result categories. Verification can:
    - PROVED: Property definitely holds (satisfiable/valid)
    - DISPROVED: Property definitely does not hold (counterexample found)
    - UNKNOWN: Solver cannot determine (incomplete theory, heuristics failed)
    - TIMEOUT: Verification exceeded time limit
    - ERROR: Internal error during verification

    References:
    - SMT-LIB standard: Standard result categories
    - Z3 solver results: sat, unsat, unknown
    """
    PROVED = "PROVED"
    DISPROVED = "DISPROVED"
    UNKNOWN = "UNKNOWN"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


@dataclass
class ProofCertificate:
    """
    A certificate attesting to a formal verification result.

    Proof certificates provide cryptographically verifiable evidence that
    a property was (or was not) proved for a specific architecture at
    a specific time with specific assumptions.

    Based on TD Commons disclosure, certificates enable:
    - Reproducibility: Same property + architecture should give same result
    - Auditability: Trace what was verified and when
    - Caching: Avoid re-proving same properties

    Attributes:
        property_hash: SHA-256 hash of the temporal property
        architecture_hash: SHA-256 hash of the architecture
        status: Verification result status
        proof_steps: List of proof steps (solver-specific format)
        assumptions: List of assumptions made during proof
        timestamp: When verification was performed (UTC)
        solver_version: Version of solver used (e.g., "z3-4.12.2")

    Example:
        >>> cert = ProofCertificate(
        ...     property_hash="abc123...",
        ...     architecture_hash="def456...",
        ...     status=VerificationStatus.PROVED,
        ...     proof_steps=[{"step": 1, "action": "simplify"}],
        ...     assumptions=["network_reliable"],
        ...     timestamp=datetime.utcnow(),
        ...     solver_version="z3-4.12.2"
        ... )
        >>> cert_hash = cert.generate_hash()

    References:
    - TD Commons: Proof certificate structure
    - Cryptographic certificates: SHA-256 for integrity
    """
    property_hash: str
    architecture_hash: str
    status: VerificationStatus
    proof_steps: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    solver_version: str = Z3_VERSION

    def generate_hash(self) -> str:
        """
        Generate SHA-256 hash of this certificate for integrity verification.

        The hash is computed over all certificate fields (except the hash itself)
        in a deterministic way to enable verification that the certificate
        has not been tampered with.

        Returns:
            Hexadecimal SHA-256 hash string

        Example:
            >>> cert = ProofCertificate(
            ...     property_hash="abc",
            ...     architecture_hash="def",
            ...     status=VerificationStatus.PROVED,
            ...     timestamp=datetime(2024, 1, 1, 12, 0, 0),
            ...     solver_version="z3-4.12.2"
            ... )
            >>> hash1 = cert.generate_hash()
            >>> hash2 = cert.generate_hash()
            >>> hash1 == hash2  # Deterministic
            True

        References:
        - SHA-256: Industry standard cryptographic hash
        - Python hashlib: https://docs.python.org/3/library/hashlib.html
        """
        cert_dict = {
            "property_hash": self.property_hash,
            "architecture_hash": self.architecture_hash,
            "status": self.status.value,
            "proof_steps": self.proof_steps,
            "assumptions": sorted(self.assumptions),
            "timestamp": self.timestamp.isoformat(),
            "solver_version": self.solver_version
        }

        # Create deterministic JSON
        json_str = json.dumps(cert_dict, sort_keys=True)

        # Compute SHA-256
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        return hash_obj.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize proof certificate to JSON-compatible dictionary.

        Returns:
            Dictionary with all certificate fields

        Example:
            >>> cert = ProofCertificate(
            ...     property_hash="abc",
            ...     architecture_hash="def",
            ...     status=VerificationStatus.PROVED,
            ...     timestamp=datetime(2024, 1, 1, 12, 0, 0),
            ...     solver_version="z3-4.12.2"
            ... )
            >>> d = cert.to_dict()
            >>> d["status"]
            'PROVED'
        """
        return {
            "property_hash": self.property_hash,
            "architecture_hash": self.architecture_hash,
            "status": self.status.value,
            "proof_steps": self.proof_steps.copy(),
            "assumptions": self.assumptions.copy(),
            "timestamp": self.timestamp.isoformat(),
            "solver_version": self.solver_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofCertificate":
        """
        Deserialize proof certificate from dictionary.

        Args:
            data: Dictionary containing certificate fields

        Returns:
            ProofCertificate instance

        Example:
            >>> data = {
            ...     "property_hash": "abc",
            ...     "architecture_hash": "def",
            ...     "status": "PROVED",
            ...     "proof_steps": [],
            ...     "assumptions": [],
            ...     "timestamp": "2024-01-01T12:00:00",
            ...     "solver_version": "z3-4.12.2"
            ... }
            >>> cert = ProofCertificate.from_dict(data)
            >>> cert.status == VerificationStatus.PROVED
            True
        """
        return cls(
            property_hash=data["property_hash"],
            architecture_hash=data["architecture_hash"],
            status=VerificationStatus(data["status"]),
            proof_steps=data.get("proof_steps", []),
            assumptions=data.get("assumptions", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            solver_version=data.get("solver_version", "unknown")
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"ProofCertificate({self.status.value}, "
            f"solver={self.solver_version})"
        )


@dataclass
class VerificationResult:
    """
    Result of verifying a temporal property against an architecture.

    Verification results capture everything needed to understand whether
    a property holds, including the property itself, verification status,
    optional proof certificate, counterexamples if disproved, and metadata.

    Based on TD Commons disclosure, verification results enable:
    - Decision making: Is this architecture safe?
    - Debugging: Why did verification fail? (counterexample)
    - Caching: Avoid re-verifying (cached flag)
    - Provenance: What was proved and when? (certificate)

    Attributes:
        property: The temporal property that was verified
        status: Verification status (PROVED, DISPROVED, etc.)
        certificate: Optional proof certificate for PROVED results
        counterexample: Optional counterexample for DISPROVED results
        execution_time: Time taken for verification (seconds)
        cached: Whether result came from cache (not freshly computed)

    Example:
        >>> result = VerificationResult(
        ...     property=TemporalProperty(...),
        ...     status=VerificationStatus.PROVED,
        ...     certificate=ProofCertificate(...),
        ...     counterexample=None,
        ...     execution_time=1.23,
        ...     cached=False
        ... )
        >>> result.verified
        True

    References:
    - TD Commons: Verification result structure
    - SMT solving: Standard result format (status + model/proof)
    """
    property: TemporalProperty
    status: VerificationStatus
    certificate: Optional[ProofCertificate] = None
    counterexample: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    cached: bool = False

    @property
    def verified(self) -> bool:
        """
        Check if property was successfully verified (proved).

        Returns:
            True if status is PROVED, False otherwise

        Example:
            >>> result = VerificationResult(
            ...     property=TemporalProperty(...),
            ...     status=VerificationStatus.PROVED,
            ...     execution_time=1.0
            ... )
            >>> result.verified
            True
            >>> result2 = VerificationResult(
            ...     property=TemporalProperty(...),
            ...     status=VerificationStatus.UNKNOWN,
            ...     execution_time=1.0
            ... )
            >>> result2.verified
            False
        """
        return self.status == VerificationStatus.PROVED

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize verification result to JSON-compatible dictionary.

        Returns:
            Dictionary with all result fields

        Example:
            >>> result = VerificationResult(
            ...     property=TemporalProperty(...),
            ...     status=VerificationStatus.PROVED,
            ...     execution_time=1.23
            ... )
            >>> d = result.to_dict()
            >>> d["status"]
            'PROVED'
        """
        return {
            "property": self.property.to_dict(),
            "status": self.status.value,
            "certificate": (
                self.certificate.to_dict() if self.certificate else None
            ),
            "counterexample": self.counterexample,
            "execution_time": self.execution_time,
            "cached": self.cached
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        """
        Deserialize verification result from dictionary.

        Args:
            data: Dictionary containing result fields

        Returns:
            VerificationResult instance

        Example:
            >>> data = {
            ...     "property": {...},
            ...     "status": "PROVED",
            ...     "certificate": None,
            ...     "counterexample": None,
            ...     "execution_time": 1.23,
            ...     "cached": False
            ... }
            >>> result = VerificationResult.from_dict(data)
            >>> result.status == VerificationStatus.PROVED
            True
        """
        return cls(
            property=TemporalProperty.from_dict(data["property"]),
            status=VerificationStatus(data["status"]),
            certificate=(
                ProofCertificate.from_dict(data["certificate"])
                if data.get("certificate") else None
            ),
            counterexample=data.get("counterexample"),
            execution_time=data.get("execution_time", 0.0),
            cached=data.get("cached", False)
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [
            f"VerificationResult({self.status.value}",
            f"property={self.property.predicate}",
        ]
        if self.cached:
            parts.append("cached=True")
        if self.execution_time > 0:
            parts.append(f"time={self.execution_time:.3f}s")

        return ", ".join(parts) + ")"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"VerificationResult(status={self.status.value}, "
            f"property={self.property.predicate}, "
            f"verified={self.verified}, "
            f"cached={self.cached})"
        )


def is_z3_available() -> bool:
    """
    Check if Z3 solver is available.

    Returns:
        True if z3-solver is installed and importable, False otherwise

    Example:
        >>> if is_z3_available():
        ...     # Use Z3 verification
        ...     pass
        ... else:
        ...     # Fall back to other verification methods
        ...     pass
    """
    return Z3_AVAILABLE


def get_z3_version() -> str:
    """
    Get Z3 solver version string.

    Returns:
        Version string (e.g., "4.12.2") or "unavailable" if not installed

    Example:
        >>> version = get_z3_version()
        >>> "4.12" in version or version == "unavailable"
        True
    """
    return Z3_VERSION
