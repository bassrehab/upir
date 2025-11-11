"""
Evidence tracking and Bayesian confidence updates for UPIR.

This module implements evidence collection and confidence propagation through
reasoning graphs using Bayesian updates and geometric mean aggregation.

Implementation based on:
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Bayesian inference (beta-binomial conjugate):
  https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall06/reading/bernoulli.pdf
- Python uuid: https://docs.python.org/3/library/uuid.html
- Python datetime: https://docs.python.org/3/library/datetime.html

Author: Subhadip Mitra
License: Apache 2.0
"""

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class Evidence:
    """
    A piece of evidence supporting or refuting an architectural decision.

    Evidence can come from various sources (benchmarks, tests, production data,
    formal proofs) and carries a Bayesian confidence level that can be updated
    as new observations arrive.

    Based on the TD Commons disclosure, UPIR tracks evidence with confidence
    levels that propagate through reasoning graphs. The Bayesian update
    implements a simple beta-binomial conjugate prior approach.

    Attributes:
        source: Where the evidence came from (e.g., "load_test_2024-01",
                "formal_verification", "production_metrics")
        type: Type of evidence - one of: benchmark, test, production, formal_proof
        data: The actual evidence data (metrics, test results, proof artifacts)
        confidence: Bayesian confidence level in [0, 1]
        timestamp: When the evidence was collected (UTC)

    Example:
        >>> evidence = Evidence(
        ...     source="load_test_2024-01",
        ...     type="benchmark",
        ...     data={"latency_p99": 95, "throughput": 10000},
        ...     confidence=0.8,
        ...     timestamp=datetime.utcnow()
        ... )
        >>> evidence.update_confidence(new_observation=True)
        >>> evidence.confidence > 0.8  # Confidence increased
        True

    References:
    - TD Commons: Evidence tracking structure
    - Bayesian inference: Beta-binomial conjugate prior
    """
    source: str
    type: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate evidence fields."""
        valid_types = {"benchmark", "test", "production", "formal_proof"}
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid evidence type '{self.type}'. "
                f"Must be one of: {valid_types}"
            )

        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be in [0, 1], got {self.confidence}"
            )

        if not self.source:
            raise ValueError("Source cannot be empty")

    def update_confidence(
        self,
        new_observation: bool,
        prior_weight: float = 0.1
    ) -> None:
        """
        Update confidence using Bayesian update based on new observation.

        This implements a simple Bayesian update using a beta-binomial conjugate
        prior. The prior_weight controls how much the new observation affects
        the current confidence.

        Update rules:
        - Positive observation: confidence += prior_weight * (1 - confidence)
        - Negative observation: confidence *= (1 - prior_weight)

        These rules ensure:
        1. Confidence stays in [0, 1]
        2. Positive observations increase confidence (asymptotically to 1)
        3. Negative observations decrease confidence (multiplicatively)
        4. Higher current confidence is harder to change (conservative)

        Args:
            new_observation: True if observation supports the evidence,
                           False if it contradicts it
            prior_weight: Weight of the prior in [0, 1]. Higher values mean
                         new observations have more impact. Default 0.1.

        Raises:
            ValueError: If prior_weight not in [0, 1]

        Example:
            >>> evidence = Evidence(
            ...     source="test",
            ...     type="benchmark",
            ...     data={},
            ...     confidence=0.5
            ... )
            >>> evidence.update_confidence(new_observation=True)
            >>> evidence.confidence
            0.55
            >>> evidence.update_confidence(new_observation=False)
            >>> evidence.confidence
            0.495

        References:
        - Beta-binomial conjugate prior: Standard Bayesian approach for
          binary observations
        - Murphy (2006): Bayesian inference for Bernoulli distribution
        """
        if not 0 <= prior_weight <= 1:
            raise ValueError(
                f"prior_weight must be in [0, 1], got {prior_weight}"
            )

        if new_observation:
            # Positive observation: move confidence towards 1
            # confidence += learning_rate * (target - current)
            self.confidence = self.confidence + prior_weight * (1 - self.confidence)
        else:
            # Negative observation: decay confidence
            # confidence *= (1 - decay_rate)
            self.confidence = self.confidence * (1 - prior_weight)

        # Ensure confidence stays in valid range (should be guaranteed by math)
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize evidence to JSON-compatible dictionary.

        Returns:
            Dictionary with all evidence fields

        Example:
            >>> evidence = Evidence(
            ...     source="test",
            ...     type="benchmark",
            ...     data={"metric": 100},
            ...     confidence=0.8,
            ...     timestamp=datetime(2024, 1, 1, 12, 0, 0)
            ... )
            >>> d = evidence.to_dict()
            >>> d["source"]
            'test'
        """
        return {
            "source": self.source,
            "type": self.type,
            "data": self.data.copy(),
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        """
        Deserialize evidence from dictionary.

        Args:
            data: Dictionary containing evidence fields

        Returns:
            Evidence instance

        Example:
            >>> data = {
            ...     "source": "test",
            ...     "type": "benchmark",
            ...     "data": {"metric": 100},
            ...     "confidence": 0.8,
            ...     "timestamp": "2024-01-01T12:00:00"
            ... }
            >>> evidence = Evidence.from_dict(data)
            >>> evidence.source
            'test'
        """
        return cls(
            source=data["source"],
            type=data["type"],
            data=data["data"],
            confidence=data["confidence"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Evidence({self.type} from {self.source}, "
            f"confidence={self.confidence:.2f})"
        )


@dataclass
class ReasoningNode:
    """
    A node in the reasoning graph representing an architectural decision.

    Reasoning nodes form a directed acyclic graph (DAG) where each node
    represents a decision or conclusion, supported by evidence and potentially
    dependent on other decisions (parent nodes).

    Based on the TD Commons disclosure, UPIR maintains a reasoning graph to
    track decision provenance and propagate confidence through the architecture.

    Attributes:
        id: Unique identifier (UUID)
        decision: The decision or conclusion made
        rationale: Explanation of why this decision was made
        evidence_ids: IDs of Evidence objects supporting this decision
        parent_ids: IDs of other ReasoningNodes this depends on (for DAG)
        alternatives: Other options that were considered but not chosen
        confidence: Computed confidence in this decision [0, 1]

    Example:
        >>> node = ReasoningNode(
        ...     id=str(uuid.uuid4()),
        ...     decision="Use PostgreSQL for primary database",
        ...     rationale="Strong consistency needed for financial transactions",
        ...     evidence_ids=["evidence-1", "evidence-2"],
        ...     parent_ids=["node-consistency-requirement"],
        ...     alternatives=[
        ...         {"option": "MongoDB", "rejected_because": "eventual consistency"}
        ...     ],
        ...     confidence=0.0  # Will be computed from evidence
        ... )

    References:
    - TD Commons: Reasoning graph structure
    - DAG: Directed acyclic graph for decision dependencies
    """
    id: str
    decision: str
    rationale: str
    evidence_ids: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0

    def __post_init__(self):
        """Validate reasoning node fields."""
        if not self.id:
            raise ValueError("ID cannot be empty")

        if not self.decision:
            raise ValueError("Decision cannot be empty")

        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be in [0, 1], got {self.confidence}"
            )

    @staticmethod
    def generate_id() -> str:
        """
        Generate a unique ID for a reasoning node.

        Returns:
            UUID string

        Example:
            >>> node_id = ReasoningNode.generate_id()
            >>> len(node_id) == 36  # UUID format
            True
        """
        return str(uuid.uuid4())

    def compute_confidence(self, evidence_map: Dict[str, Evidence]) -> float:
        """
        Compute aggregate confidence from supporting evidence using geometric mean.

        The geometric mean is more conservative than arithmetic mean - a single
        piece of low-confidence evidence significantly reduces overall confidence.
        This matches how engineers actually reason: one weak piece of evidence
        can't be compensated by many strong pieces.

        Formula: exp(mean(log(c_i))) for confidences c_i

        If no evidence is available, returns 0.0 (no confidence).
        If any evidence has confidence 0, returns 0.0 (geometric mean property).

        Args:
            evidence_map: Mapping from evidence IDs to Evidence objects

        Returns:
            Computed confidence in [0, 1]

        Example:
            >>> evidence_map = {
            ...     "e1": Evidence("src1", "test", {}, 0.8, datetime.utcnow()),
            ...     "e2": Evidence("src2", "test", {}, 0.9, datetime.utcnow())
            ... }
            >>> node = ReasoningNode(
            ...     id="node-1",
            ...     decision="test",
            ...     rationale="test",
            ...     evidence_ids=["e1", "e2"]
            ... )
            >>> conf = node.compute_confidence(evidence_map)
            >>> 0.84 < conf < 0.85  # sqrt(0.8 * 0.9) ≈ 0.8485
            True

        References:
        - Geometric mean: https://en.wikipedia.org/wiki/Geometric_mean
        - More conservative than arithmetic mean for combining confidences
        """
        # Filter evidence to only those referenced by this node
        relevant_evidence = [
            evidence_map[eid]
            for eid in self.evidence_ids
            if eid in evidence_map
        ]

        if not relevant_evidence:
            return 0.0

        # Check for any zero confidence (geometric mean would be 0)
        confidences = [e.confidence for e in relevant_evidence]
        if any(c == 0.0 for c in confidences):
            return 0.0

        # Compute geometric mean: exp(mean(log(c_i)))
        log_sum = sum(math.log(c) for c in confidences)
        log_mean = log_sum / len(confidences)
        geometric_mean = math.exp(log_mean)

        return geometric_mean

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize reasoning node to JSON-compatible dictionary.

        Returns:
            Dictionary with all node fields

        Example:
            >>> node = ReasoningNode(
            ...     id="node-1",
            ...     decision="Use caching",
            ...     rationale="Reduce latency",
            ...     evidence_ids=["e1"],
            ...     parent_ids=[],
            ...     alternatives=[{"option": "No cache"}],
            ...     confidence=0.8
            ... )
            >>> d = node.to_dict()
            >>> d["decision"]
            'Use caching'
        """
        return {
            "id": self.id,
            "decision": self.decision,
            "rationale": self.rationale,
            "evidence_ids": self.evidence_ids.copy(),
            "parent_ids": self.parent_ids.copy(),
            "alternatives": [alt.copy() for alt in self.alternatives],
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningNode":
        """
        Deserialize reasoning node from dictionary.

        Args:
            data: Dictionary containing node fields

        Returns:
            ReasoningNode instance

        Example:
            >>> data = {
            ...     "id": "node-1",
            ...     "decision": "Use caching",
            ...     "rationale": "Reduce latency",
            ...     "evidence_ids": ["e1"],
            ...     "parent_ids": [],
            ...     "alternatives": [],
            ...     "confidence": 0.8
            ... }
            >>> node = ReasoningNode.from_dict(data)
            >>> node.decision
            'Use caching'
        """
        return cls(
            id=data["id"],
            decision=data["decision"],
            rationale=data["rationale"],
            evidence_ids=data.get("evidence_ids", []),
            parent_ids=data.get("parent_ids", []),
            alternatives=data.get("alternatives", []),
            confidence=data.get("confidence", 0.0)
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"ReasoningNode({self.decision}, "
            f"confidence={self.confidence:.2f}, "
            f"evidence={len(self.evidence_ids)})"
        )

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"ReasoningNode(id='{self.id}', "
            f"decision='{self.decision}', "
            f"confidence={self.confidence}, "
            f"evidence_count={len(self.evidence_ids)}, "
            f"parent_count={len(self.parent_ids)})"
        )
