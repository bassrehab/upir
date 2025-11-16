"""
Main UPIR class integrating specification, evidence, reasoning, and architecture.

This module provides the core UPIR (Universal Plan Intermediate Representation)
class that ties together all components: formal specifications, evidence tracking,
reasoning graphs, and architecture representations.

Implementation based on:
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Graph algorithms (cycle detection): Standard DFS with recursion stack
- Python datetime: https://docs.python.org/3/library/datetime.html
- Python hashlib: https://docs.python.org/3/library/hashlib.html

Author: Subhadip Mitra
License: Apache 2.0
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Set

from upir.core.architecture import Architecture
from upir.core.evidence import Evidence, ReasoningNode
from upir.core.specification import FormalSpecification


@dataclass
class UPIR:
    """
    Universal Plan Intermediate Representation for distributed systems.

    UPIR integrates formal specifications, architectural designs, evidence
    from various sources, and reasoning graphs to enable automated synthesis,
    verification, and continuous optimization of distributed systems.

    Based on TD Commons disclosure, UPIR maintains:
    - Formal specifications (invariants, properties, constraints)
    - Architecture representation (components, connections, deployment)
    - Evidence tracking (benchmarks, tests, production data, proofs)
    - Reasoning graph (decisions, rationale, confidence propagation)

    Attributes:
        specification: Formal specification with invariants and properties
        architecture: Architecture representation (components, connections)
        id: Unique identifier (auto-generated UUID if not provided)
        name: Human-readable name for this UPIR instance (default: "")
        description: Description of the system being represented (default: "")
        evidence: Dictionary of evidence keyed by evidence ID
        reasoning: Dictionary of reasoning nodes keyed by node ID
        metadata: Additional metadata (tags, owner, project, etc.)
        created_at: When this UPIR was created (UTC)
        updated_at: When this UPIR was last modified (UTC)

    Example:
        >>> # Simple usage with defaults
        >>> upir = UPIR(
        ...     specification=FormalSpecification(...),
        ...     architecture=Architecture(...)
        ... )
        >>> # Or with explicit metadata
        >>> upir = UPIR(
        ...     specification=spec,
        ...     architecture=arch,
        ...     name="E-commerce Platform",
        ...     description="High-throughput e-commerce system"
        ... )
        >>> upir.validate()
        True

    References:
    - TD Commons: UPIR structure and components
    """
    specification: Optional[FormalSpecification] = None
    architecture: Optional[Architecture] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    evidence: Dict[str, Evidence] = field(default_factory=dict)
    reasoning: Dict[str, ReasoningNode] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate UPIR fields."""
        if not self.id:
            raise ValueError("ID cannot be empty")

    @staticmethod
    def generate_id() -> str:
        """
        Generate a unique ID for a UPIR instance.

        Returns:
            UUID string

        Example:
            >>> upir_id = UPIR.generate_id()
            >>> len(upir_id) == 36  # UUID format
            True
        """
        return str(uuid.uuid4())

    def add_evidence(self, evidence: Evidence) -> str:
        """
        Add evidence to this UPIR instance.

        Generates a unique ID for the evidence and adds it to the evidence
        dictionary. Updates the updated_at timestamp.

        Args:
            evidence: Evidence instance to add

        Returns:
            Generated evidence ID

        Example:
            >>> upir = UPIR(id="upir-1", name="test", description="test")
            >>> evidence = Evidence("test", "benchmark", {}, 0.8)
            >>> evidence_id = upir.add_evidence(evidence)
            >>> len(evidence_id) == 36  # UUID
            True
            >>> evidence_id in upir.evidence
            True
        """
        evidence_id = str(uuid.uuid4())
        self.evidence[evidence_id] = evidence
        self.updated_at = datetime.utcnow()
        return evidence_id

    def add_reasoning(self, node: ReasoningNode) -> str:
        """
        Add reasoning node to this UPIR instance.

        Adds the node to the reasoning dictionary using its ID. Updates the
        updated_at timestamp.

        Args:
            node: ReasoningNode instance to add

        Returns:
            The node's ID (from node.id)

        Example:
            >>> upir = UPIR(id="upir-1", name="test", description="test")
            >>> node = ReasoningNode(
            ...     id=ReasoningNode.generate_id(),
            ...     decision="Use PostgreSQL",
            ...     rationale="Strong consistency needed"
            ... )
            >>> node_id = upir.add_reasoning(node)
            >>> node_id in upir.reasoning
            True
        """
        self.reasoning[node.id] = node
        self.updated_at = datetime.utcnow()
        return node.id

    def compute_overall_confidence(self) -> float:
        """
        Compute overall confidence using harmonic mean of leaf node confidences.

        Leaf nodes are reasoning nodes that are not referenced as parents by
        any other nodes - they represent final decisions or conclusions.

        The harmonic mean is more conservative than arithmetic or geometric mean,
        heavily penalizing low confidences. Formula: n / sum(1/c_i)

        Returns:
            Overall confidence in [0, 1], or 0.0 if no leaf nodes

        Example:
            >>> upir = UPIR(id="upir-1", name="test", description="test")
            >>> node1 = ReasoningNode("n1", "decision1", "rationale", confidence=0.8)
            >>> node2 = ReasoningNode("n2", "decision2", "rationale", confidence=0.9)
            >>> upir.add_reasoning(node1)
            'n1'
            >>> upir.add_reasoning(node2)
            'n2'
            >>> conf = upir.compute_overall_confidence()
            >>> 0.84 < conf < 0.85  # Harmonic mean of 0.8 and 0.9
            True

        References:
        - Harmonic mean: More conservative aggregation for confidences
        - Leaf nodes: Final conclusions in reasoning DAG
        """
        if not self.reasoning:
            return 0.0

        # Find all nodes referenced as parents
        referenced_nodes: Set[str] = set()
        for node in self.reasoning.values():
            referenced_nodes.update(node.parent_ids)

        # Leaf nodes are those NOT referenced as parents
        leaf_nodes = [
            node for node_id, node in self.reasoning.items()
            if node_id not in referenced_nodes
        ]

        if not leaf_nodes:
            return 0.0

        # Filter out zero confidences (would cause division by zero)
        confidences = [node.confidence for node in leaf_nodes]
        if any(c == 0.0 for c in confidences):
            return 0.0

        # Compute harmonic mean: n / sum(1/c_i)
        n = len(confidences)
        reciprocal_sum = sum(1.0 / c for c in confidences)
        harmonic_mean = n / reciprocal_sum

        return harmonic_mean

    def validate(self) -> bool:
        """
        Validate the UPIR instance for consistency.

        Performs multiple validation checks:
        1. Specification validation (if present)
        2. Reasoning DAG cycle detection
        3. Evidence reference validation (all referenced evidence exists)

        Returns:
            True if all validations pass

        Raises:
            ValueError: If any validation check fails

        Example:
            >>> upir = UPIR(
            ...     id="upir-1",
            ...     name="test",
            ...     description="test",
            ...     specification=FormalSpecification()
            ... )
            >>> upir.validate()
            True

        References:
        - Cycle detection: DFS with recursion stack
        - Standard graph algorithm for DAG validation
        """
        # Validate specification if present
        if self.specification is not None:
            self.specification.validate()

        # Check for cycles in reasoning DAG
        self._check_dag_cycles()

        # Validate evidence references
        for node in self.reasoning.values():
            for evidence_id in node.evidence_ids:
                if evidence_id not in self.evidence:
                    raise ValueError(
                        f"Reasoning node '{node.id}' references non-existent "
                        f"evidence '{evidence_id}'"
                    )

        return True

    def _check_dag_cycles(self) -> None:
        """
        Check for cycles in the reasoning DAG using DFS.

        Uses depth-first search with a recursion stack to detect cycles.
        If a node is encountered that's already in the recursion stack,
        a cycle exists.

        Raises:
            ValueError: If a cycle is detected

        Algorithm:
        - visited: Set of all nodes we've completely processed
        - rec_stack: Set of nodes in current DFS path (recursion stack)
        - For each unvisited node, do DFS
        - If we visit a node already in rec_stack -> cycle

        References:
        - Standard DFS cycle detection for directed graphs
        - Cormen et al., "Introduction to Algorithms"
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node_id: str) -> None:
            """DFS helper to detect cycles."""
            visited.add(node_id)
            rec_stack.add(node_id)

            # Visit all parent nodes
            if node_id in self.reasoning:
                node = self.reasoning[node_id]
                for parent_id in node.parent_ids:
                    if parent_id not in visited:
                        # Recursively visit parent
                        dfs(parent_id)
                    elif parent_id in rec_stack:
                        # Found a back edge -> cycle
                        raise ValueError(
                            f"Cycle detected in reasoning graph involving "
                            f"node '{parent_id}'"
                        )

            # Remove from recursion stack when done
            rec_stack.remove(node_id)

        # Check all nodes
        for node_id in self.reasoning.keys():
            if node_id not in visited:
                dfs(node_id)

    def generate_signature(self) -> str:
        """
        Generate cryptographic signature (SHA-256) of this UPIR.

        Creates a deterministic hash of the entire UPIR state including
        specification, architecture, evidence, and reasoning. Useful for
        integrity verification and change detection.

        Returns:
            Hexadecimal SHA-256 hash string

        Example:
            >>> upir = UPIR(id="upir-1", name="test", description="test")
            >>> sig1 = upir.generate_signature()
            >>> sig2 = upir.generate_signature()
            >>> sig1 == sig2  # Deterministic
            True
            >>> len(sig1) == 64  # SHA-256
            True

        References:
        - SHA-256: Industry standard cryptographic hash
        - Python hashlib: https://docs.python.org/3/library/hashlib.html
        """
        # Convert to dict representation
        upir_dict = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "specification": (
                self.specification.to_dict() if self.specification else None
            ),
            "architecture": (
                self.architecture.to_dict() if self.architecture else None
            ),
            "evidence": {
                eid: ev.to_dict() for eid, ev in sorted(self.evidence.items())
            },
            "reasoning": {
                nid: node.to_dict() for nid, node in sorted(self.reasoning.items())
            },
            "metadata": dict(sorted(self.metadata.items()))
        }

        # Create deterministic JSON
        json_str = json.dumps(upir_dict, sort_keys=True)

        # Compute SHA-256
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        return hash_obj.hexdigest()

    def to_json(self) -> str:
        """
        Serialize UPIR to JSON string.

        Returns:
            JSON string representation

        Example:
            >>> upir = UPIR(id="upir-1", name="test", description="desc")
            >>> json_str = upir.to_json()
            >>> "upir-1" in json_str
            True
        """
        upir_dict = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "specification": (
                self.specification.to_dict() if self.specification else None
            ),
            "architecture": (
                self.architecture.to_dict() if self.architecture else None
            ),
            "evidence": {
                eid: ev.to_dict() for eid, ev in self.evidence.items()
            },
            "reasoning": {
                nid: node.to_dict() for nid, node in self.reasoning.items()
            },
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        return json.dumps(upir_dict, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "UPIR":
        """
        Deserialize UPIR from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            UPIR instance

        Example:
            >>> original = UPIR(id="upir-1", name="test", description="desc")
            >>> json_str = original.to_json()
            >>> restored = UPIR.from_json(json_str)
            >>> restored.id == original.id
            True
        """
        data = json.loads(json_str)

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            specification=(
                FormalSpecification.from_dict(data["specification"])
                if data.get("specification") else None
            ),
            architecture=(
                Architecture.from_dict(data["architecture"])
                if data.get("architecture") else None
            ),
            evidence={
                eid: Evidence.from_dict(ev_data)
                for eid, ev_data in data.get("evidence", {}).items()
            },
            reasoning={
                nid: ReasoningNode.from_dict(node_data)
                for nid, node_data in data.get("reasoning", {}).items()
            },
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [f"'{self.name}'"]
        if self.specification:
            parts.append("with spec")
        if self.architecture:
            parts.append("with arch")
        if self.evidence:
            parts.append(f"{len(self.evidence)} evidence")
        if self.reasoning:
            parts.append(f"{len(self.reasoning)} reasoning nodes")

        return f"UPIR({', '.join(parts)})"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"UPIR(id='{self.id}', name='{self.name}', "
            f"spec={self.specification is not None}, "
            f"arch={self.architecture is not None}, "
            f"evidence={len(self.evidence)}, "
            f"reasoning={len(self.reasoning)})"
        )
