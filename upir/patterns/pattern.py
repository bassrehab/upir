"""
Architectural pattern representation.

This module defines the Pattern dataclass for representing discovered or
predefined architectural patterns that can be reused across multiple UPIRs.

Implementation based on:
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Pattern recognition and template design

Author: Subhadip Mitra
License: Apache 2.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Pattern:
    """
    An architectural pattern discovered from or applied to UPIRs.

    Patterns represent common architectural structures that appear across
    multiple UPIR instances. They can be discovered through clustering or
    defined manually as templates.

    A pattern includes a template structure, metadata about instances that
    match the pattern, and performance metrics for pattern effectiveness.

    Attributes:
        id: Unique identifier for the pattern
        name: Human-readable name (e.g., "streaming-etl", "api-gateway")
        description: Detailed description of the pattern
        template: Template structure with parameterizable components
                 Format: {
                     "components": [{"type": "...", "properties": {...}}],
                     "connections": [{"from": "...", "to": "...", ...}],
                     "parameters": {...}  # Tunable parameters
                 }
        instances: List of UPIR IDs that match this pattern
        success_rate: Fraction of instances meeting their specifications (0-1)
        average_performance: Average metrics across instances
                            Format: {"latency_p99": ..., "throughput_qps": ...}

    Example:
        >>> pattern = Pattern(
        ...     id="streaming-etl-1",
        ...     name="Streaming ETL Pipeline",
        ...     description="Event-driven data processing pipeline",
        ...     template={
        ...         "components": [
        ...             {"type": "pubsub_source", "properties": {}},
        ...             {"type": "stream_processor", "properties": {}},
        ...             {"type": "bigquery_sink", "properties": {}}
        ...         ],
        ...         "parameters": {"window_size": 60, "parallelism": 10}
        ...     },
        ...     instances=["upir-1", "upir-2", "upir-3"],
        ...     success_rate=0.95,
        ...     average_performance={"latency_p99": 150, "throughput_qps": 5000}
        ... )

    References:
    - TD Commons: Pattern extraction and reuse
    - Design patterns: Template method pattern
    """

    id: str
    name: str
    description: str
    template: Dict[str, Any]
    instances: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    average_performance: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate pattern fields."""
        if not self.id:
            raise ValueError("Pattern id cannot be empty")
        if not self.name:
            raise ValueError("Pattern name cannot be empty")
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError(f"Success rate must be in [0, 1], got {self.success_rate}")

    def add_instance(self, upir_id: str, performance: Dict[str, float] = None):
        """
        Add a UPIR instance to this pattern.

        Updates instances list and recalculates average performance.

        Args:
            upir_id: ID of the UPIR instance
            performance: Performance metrics for this instance
        """
        if upir_id not in self.instances:
            self.instances.append(upir_id)

        # Update average performance if metrics provided
        if performance:
            for metric, value in performance.items():
                if metric not in self.average_performance:
                    self.average_performance[metric] = value
                else:
                    # Incremental average: new_avg = old_avg + (new_val - old_avg) / count
                    count = len(self.instances)
                    old_avg = self.average_performance[metric]
                    self.average_performance[metric] = old_avg + (value - old_avg) / count

    def matches(self, feature_vector: List[float], threshold: float = 0.8) -> bool:
        """
        Check if a feature vector matches this pattern.

        Uses cosine similarity between the feature vector and the pattern's
        centroid (if available in template).

        Args:
            feature_vector: Feature vector to check
            threshold: Similarity threshold (0-1)

        Returns:
            True if similarity >= threshold
        """
        if "centroid" not in self.template:
            return False

        import numpy as np
        centroid = np.array(self.template["centroid"])
        vector = np.array(feature_vector)

        # Cosine similarity
        dot_product = np.dot(centroid, vector)
        norm_product = np.linalg.norm(centroid) * np.linalg.norm(vector)

        if norm_product == 0:
            return False

        similarity = dot_product / norm_product
        return similarity >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize pattern to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "instances": self.instances,
            "success_rate": self.success_rate,
            "average_performance": self.average_performance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """
        Deserialize pattern from dictionary.

        Args:
            data: Dictionary containing pattern fields

        Returns:
            Pattern instance
        """
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            template=data.get("template", {}),
            instances=data.get("instances", []),
            success_rate=data.get("success_rate", 0.0),
            average_performance=data.get("average_performance", {}),
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Pattern({self.name}, "
            f"{len(self.instances)} instances, "
            f"success_rate={self.success_rate:.2f})"
        )

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"Pattern(id='{self.id}', name='{self.name}', "
            f"instances={len(self.instances)}, "
            f"success_rate={self.success_rate:.2f})"
        )
