"""
Temporal property system for UPIR formal verification.

This module implements temporal properties using Linear Temporal Logic (LTL)
operators for expressing time-dependent constraints on distributed systems.

Implementation based on:
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Temporal Logic: Pnueli, A. (1977) "The temporal logic of programs"
  https://www.cs.toronto.edu/~hehner/FMCO/Pnueli.pdf
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
- Z3 SMT encoding: https://microsoft.github.io/z3guide/

Author: Subhadip Mitra
License: Apache 2.0
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TemporalOperator(Enum):
    """
    Linear Temporal Logic (LTL) operators for expressing temporal properties.

    Based on Pnueli's temporal logic formalism, these operators allow
    expressing properties about all time points (ALWAYS), some future time
    point (EVENTUALLY), bounded time constraints (WITHIN), and sequential
    properties (UNTIL).

    References:
    - Pnueli (1977): Standard LTL operators
    - TD Commons disclosure: UPIR temporal property specification
    """
    ALWAYS = "ALWAYS"  # ◻P or G(P) - Property holds at all time points
    EVENTUALLY = "EVENTUALLY"  # ◇P or F(P) - Property holds at some future point
    WITHIN = "WITHIN"  # Bounded eventually - Property holds within time bound
    UNTIL = "UNTIL"  # P U Q - P holds until Q becomes true


@dataclass
class TemporalProperty:
    """
    A temporal property with formal semantics for distributed system verification.

    Temporal properties express constraints that must hold over time, such as
    "the system ALWAYS responds to requests" or "backups complete WITHIN 1 hour".

    Based on Linear Temporal Logic (LTL) as defined in Pnueli (1977), extended
    with bounded operators for practical system verification per TD Commons.

    Attributes:
        operator: The temporal operator (ALWAYS, EVENTUALLY, WITHIN, UNTIL)
        predicate: String description of the property being asserted
                  (e.g., "data_consistent", "request_processed")
        time_bound: Optional time bound in seconds for bounded operators (WITHIN)
        parameters: Additional parameters for property evaluation (e.g., thresholds)

    Example:
        >>> # System always responds within 100ms
        >>> prop = TemporalProperty(
        ...     operator=TemporalOperator.WITHIN,
        ...     predicate="response_received",
        ...     time_bound=0.1,
        ...     parameters={"max_latency_ms": 100}
        ... )

    References:
    - TD Commons: Temporal property structure
    - Pnueli (1977): LTL semantics
    """
    operator: TemporalOperator
    predicate: str
    time_bound: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate temporal property constraints."""
        if self.operator == TemporalOperator.WITHIN and self.time_bound is None:
            raise ValueError("WITHIN operator requires a time_bound")

        if self.time_bound is not None and self.time_bound <= 0:
            raise ValueError("time_bound must be positive")

        if not self.predicate:
            raise ValueError("predicate cannot be empty")

    def to_smt(self) -> str:
        """
        Convert temporal property to SMT-LIB format for Z3 solver.

        Encoding follows standard temporal logic to first-order logic translation:
        - ALWAYS P: ∀t. P(t) - Universal quantification over time
        - EVENTUALLY P: ∃t. P(t) - Existential quantification over time
        - WITHIN P (bound b): ∃t. (t ≤ b) ∧ P(t) - Bounded existential
        - P UNTIL Q: ∃t. Q(t) ∧ ∀s. (s < t) → P(s) - Q eventually holds, P until then

        Returns:
            SMT-LIB formatted string suitable for Z3

        References:
        - Z3 tutorial: https://microsoft.github.io/z3guide/
        - Temporal logic encoding: Standard translation to FOL

        Example:
            >>> prop = TemporalProperty(
            ...     operator=TemporalOperator.ALWAYS,
            ...     predicate="data_consistent"
            ... )
            >>> smt = prop.to_smt()
            >>> "forall" in smt
            True
        """
        pred_name = self.predicate

        if self.operator == TemporalOperator.ALWAYS:
            # ∀t. P(t) - Property holds at all time points
            return f"(forall ((t Real)) ({pred_name} t))"

        elif self.operator == TemporalOperator.EVENTUALLY:
            # ∃t. P(t) - Property holds at some future time point
            return f"(exists ((t Real)) ({pred_name} t))"

        elif self.operator == TemporalOperator.WITHIN:
            # ∃t. (t ≤ bound) ∧ P(t) - Property holds within time bound
            bound = self.time_bound
            return f"(exists ((t Real)) (and (<= t {bound}) ({pred_name} t)))"

        elif self.operator == TemporalOperator.UNTIL:
            # P UNTIL Q encoding: ∃t. Q(t) ∧ ∀s. (s < t) → P(s)
            # Note: For UNTIL, predicate should contain both P and Q predicates
            # This is a simplified encoding; full UNTIL may need parameters
            return (
                f"(exists ((t Real)) "
                f"(and ({pred_name}_q t) "
                f"(forall ((s Real)) (=> (< s t) ({pred_name}_p s)))))"
            )

        else:
            raise ValueError(f"Unknown temporal operator: {self.operator}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize temporal property to JSON-compatible dictionary.

        Returns:
            Dictionary with all property fields in serializable format

        Example:
            >>> prop = TemporalProperty(
            ...     operator=TemporalOperator.WITHIN,
            ...     predicate="backup_complete",
            ...     time_bound=3600.0,
            ...     parameters={"backup_type": "full"}
            ... )
            >>> d = prop.to_dict()
            >>> d["operator"]
            'WITHIN'
        """
        return {
            "operator": self.operator.value,
            "predicate": self.predicate,
            "time_bound": self.time_bound,
            "parameters": self.parameters.copy()
        }

    def hash(self) -> str:
        """
        Generate SHA-256 hash of this temporal property.

        Uses deterministic JSON serialization to ensure same property
        always produces the same hash, enabling caching and integrity checks.

        Returns:
            Hexadecimal SHA-256 hash string

        Example:
            >>> prop = TemporalProperty(
            ...     operator=TemporalOperator.ALWAYS,
            ...     predicate="test"
            ... )
            >>> hash1 = prop.hash()
            >>> hash2 = prop.hash()
            >>> hash1 == hash2  # Deterministic
            True

        References:
        - SHA-256: Industry standard cryptographic hash
        - Python hashlib: https://docs.python.org/3/library/hashlib.html
        """
        prop_dict = self.to_dict()
        json_str = json.dumps(prop_dict, sort_keys=True)
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        return hash_obj.hexdigest()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalProperty":
        """
        Deserialize temporal property from dictionary.

        Args:
            data: Dictionary containing property fields

        Returns:
            TemporalProperty instance

        Raises:
            ValueError: If operator is invalid or required fields missing

        Example:
            >>> data = {
            ...     "operator": "ALWAYS",
            ...     "predicate": "data_consistent",
            ...     "time_bound": None,
            ...     "parameters": {}
            ... }
            >>> prop = TemporalProperty.from_dict(data)
            >>> prop.operator == TemporalOperator.ALWAYS
            True
        """
        return cls(
            operator=TemporalOperator(data["operator"]),
            predicate=data["predicate"],
            time_bound=data.get("time_bound"),
            parameters=data.get("parameters", {})
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.time_bound is not None:
            return f"{self.operator.value}[{self.time_bound}s]({self.predicate})"
        return f"{self.operator.value}({self.predicate})"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"TemporalProperty(operator={self.operator}, "
            f"predicate='{self.predicate}', "
            f"time_bound={self.time_bound}, "
            f"parameters={self.parameters})"
        )
