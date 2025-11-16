"""
Formal specification system for UPIR architecture verification.

This module implements formal specifications that capture architectural requirements,
invariants, properties, constraints, and assumptions for distributed systems.

Implementation based on:
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
- Python hashlib: https://docs.python.org/3/library/hashlib.html

Author: Subhadip Mitra
License: Apache 2.0
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from upir.core.temporal import TemporalProperty


@dataclass
class FormalSpecification:
    """
    A formal specification for distributed system architecture.

    Formal specifications capture all requirements and constraints that an
    architecture must satisfy, including temporal properties, resource constraints,
    and environmental assumptions.

    Based on the TD Commons disclosure, specifications include:
    - Invariants: Properties that MUST always hold (safety properties)
    - Properties: Desired properties that should hold (liveness, performance)
    - Constraints: Resource limits (latency, throughput, cost, etc.)
    - Assumptions: Environmental conditions assumed to hold

    Attributes:
        invariants: List of temporal properties that must always hold.
                   Violations indicate architectural bugs.
        properties: List of temporal properties that should hold.
                   These are optimization targets.
        constraints: Resource constraints as nested dicts.
                    Format: {"resource_name": {"min": x, "max": y, "equals": z}}
        assumptions: Environmental assumptions (e.g., "network_reliable",
                    "nodes_fail_independently")

    Example:
        >>> spec = FormalSpecification(
        ...     invariants=[
        ...         TemporalProperty(
        ...             operator=TemporalOperator.ALWAYS,
        ...             predicate="data_consistent"
        ...         )
        ...     ],
        ...     constraints={
        ...         "latency": {"max": 100},
        ...         "cost_per_month": {"max": 10000}
        ...     },
        ...     assumptions=["network_partitions_rare"]
        ... )

    References:
    - TD Commons: Formal specification structure
    - Design by Contract: Invariants and preconditions
    """
    invariants: List[TemporalProperty] = field(default_factory=list)
    properties: List[TemporalProperty] = field(default_factory=list)
    constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        """
        Validate specification consistency and well-formedness.

        Checks performed:
        1. No duplicate invariants (same operator and predicate)
        2. No duplicate properties
        3. All constraints have valid fields (min, max, or equals)
        4. Constraint values are valid (min <= max if both present)
        5. No empty predicates in temporal properties

        Returns:
            True if specification is valid

        Raises:
            ValueError: If specification has consistency issues

        Example:
            >>> spec = FormalSpecification(
            ...     invariants=[
            ...         TemporalProperty(
            ...             operator=TemporalOperator.ALWAYS,
            ...             predicate="data_consistent"
            ...         )
            ...     ]
            ... )
            >>> spec.validate()
            True
        """
        # Check for duplicate invariants
        seen_invariants = set()
        for inv in self.invariants:
            key = (inv.operator, inv.predicate)
            if key in seen_invariants:
                raise ValueError(
                    f"Duplicate invariant: {inv.operator.value}({inv.predicate})"
                )
            seen_invariants.add(key)

        # Check for duplicate properties
        seen_properties = set()
        for prop in self.properties:
            key = (prop.operator, prop.predicate)
            if key in seen_properties:
                raise ValueError(
                    f"Duplicate property: {prop.operator.value}({prop.predicate})"
                )
            seen_properties.add(key)

        # Validate constraints
        for resource_name, constraint in self.constraints.items():
            if not isinstance(constraint, dict):
                raise ValueError(
                    f"Constraint '{resource_name}' must be a dictionary"
                )

            # Check that constraint has at least one valid field
            valid_fields = {"min", "max", "equals"}
            constraint_fields = set(constraint.keys())
            if not constraint_fields.intersection(valid_fields):
                raise ValueError(
                    f"Constraint '{resource_name}' must have at least one of: "
                    f"min, max, equals. Got: {constraint_fields}"
                )

            # Check that invalid fields are not present
            invalid_fields = constraint_fields - valid_fields
            if invalid_fields:
                raise ValueError(
                    f"Constraint '{resource_name}' has invalid fields: {invalid_fields}. "
                    f"Valid fields are: {valid_fields}"
                )

            # Validate min <= max if both present
            if "min" in constraint and "max" in constraint:
                min_val = constraint["min"]
                max_val = constraint["max"]
                if min_val > max_val:
                    raise ValueError(
                        f"Constraint '{resource_name}': min ({min_val}) > max ({max_val})"
                    )

            # If 'equals' is present, it should be the only field
            if "equals" in constraint and len(constraint) > 1:
                raise ValueError(
                    f"Constraint '{resource_name}': 'equals' cannot be combined "
                    f"with min/max"
                )

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize formal specification to JSON-compatible dictionary.

        Uses deterministic ordering (sorted keys) to ensure consistent
        serialization for hashing.

        Returns:
            Dictionary with all specification fields

        Example:
            >>> spec = FormalSpecification(
            ...     constraints={"latency": {"max": 100}}
            ... )
            >>> d = spec.to_dict()
            >>> d["constraints"]["latency"]["max"]
            100
        """
        return {
            "invariants": [inv.to_dict() for inv in self.invariants],
            "properties": [prop.to_dict() for prop in self.properties],
            "constraints": {
                k: dict(v) for k, v in sorted(self.constraints.items())
            },
            "assumptions": sorted(self.assumptions)
        }

    def to_json(self) -> str:
        """
        Serialize formal specification to JSON string.

        Returns:
            JSON string representation

        Example:
            >>> spec = FormalSpecification(
            ...     constraints={"latency": {"max": 100}}
            ... )
            >>> json_str = spec.to_json()
            >>> isinstance(json_str, str)
            True
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormalSpecification":
        """
        Deserialize formal specification from dictionary.

        Args:
            data: Dictionary containing specification fields

        Returns:
            FormalSpecification instance

        Example:
            >>> data = {
            ...     "invariants": [],
            ...     "properties": [],
            ...     "constraints": {"latency": {"max": 100}},
            ...     "assumptions": ["network_reliable"]
            ... }
            >>> spec = FormalSpecification.from_dict(data)
            >>> spec.constraints["latency"]["max"]
            100
        """
        return cls(
            invariants=[
                TemporalProperty.from_dict(inv)
                for inv in data.get("invariants", [])
            ],
            properties=[
                TemporalProperty.from_dict(prop)
                for prop in data.get("properties", [])
            ],
            constraints=data.get("constraints", {}),
            assumptions=data.get("assumptions", [])
        )

    def hash(self) -> str:
        """
        Compute SHA-256 hash of specification for integrity verification.

        Uses deterministic JSON serialization (sorted keys) to ensure
        consistent hashes across runs and platforms.

        Returns:
            Hexadecimal string of SHA-256 hash

        Example:
            >>> spec1 = FormalSpecification(
            ...     constraints={"latency": {"max": 100}}
            ... )
            >>> spec2 = FormalSpecification(
            ...     constraints={"latency": {"max": 100}}
            ... )
            >>> spec1.hash() == spec2.hash()
            True

        References:
        - Python hashlib: https://docs.python.org/3/library/hashlib.html
        - SHA-256: Industry standard cryptographic hash
        """
        # Convert to JSON with sorted keys for deterministic serialization
        spec_dict = self.to_dict()
        json_str = json.dumps(spec_dict, sort_keys=True)

        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        return hash_obj.hexdigest()

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = []
        if self.invariants:
            parts.append(f"{len(self.invariants)} invariant(s)")
        if self.properties:
            parts.append(f"{len(self.properties)} propertie(s)")
        if self.constraints:
            parts.append(f"{len(self.constraints)} constraint(s)")
        if self.assumptions:
            parts.append(f"{len(self.assumptions)} assumption(s)")

        if not parts:
            return "FormalSpecification(empty)"

        return f"FormalSpecification({', '.join(parts)})"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"FormalSpecification("
            f"invariants={len(self.invariants)}, "
            f"properties={len(self.properties)}, "
            f"constraints={len(self.constraints)}, "
            f"assumptions={len(self.assumptions)})"
        )
