"""
Simple architecture representation for UPIR.

This module provides a placeholder architecture structure for representing
distributed system architectures. Future versions will expand this with
richer modeling capabilities.

Implementation based on:
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html

Author: Subhadip Mitra
License: Apache 2.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Architecture:
    """
    A simple representation of a distributed system architecture.

    This is a placeholder structure that will be expanded in future versions
    with richer modeling capabilities, component templates, and validation.

    Based on TD Commons disclosure, architectures consist of components
    (services, databases, caches), connections (network links, APIs),
    deployment configuration, and applied patterns.

    Attributes:
        components: List of architectural components (services, DBs, etc.)
                   Each component is a dict with keys like: name, type, config
        connections: List of connections between components
                    Each connection is a dict with: from, to, protocol, etc.
        deployment: Deployment configuration (regions, resources, scaling)
        patterns: List of architectural patterns applied (e.g., "CQRS", "event-sourcing")

    Example:
        >>> arch = Architecture(
        ...     components=[
        ...         {"name": "api-service", "type": "service", "replicas": 3},
        ...         {"name": "postgres", "type": "database", "size": "large"}
        ...     ],
        ...     connections=[
        ...         {"from": "api-service", "to": "postgres", "protocol": "TCP"}
        ...     ],
        ...     deployment={
        ...         "regions": ["us-west-2", "us-east-1"],
        ...         "strategy": "blue-green"
        ...     },
        ...     patterns=["microservices", "CQRS"]
        ... )

    References:
    - TD Commons: Architecture representation
    """
    components: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    deployment: Dict[str, Any] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize architecture to JSON-compatible dictionary.

        Returns:
            Dictionary with all architecture fields

        Example:
            >>> arch = Architecture(
            ...     components=[{"name": "service"}],
            ...     patterns=["microservices"]
            ... )
            >>> d = arch.to_dict()
            >>> d["patterns"]
            ['microservices']
        """
        return {
            "components": [comp.copy() for comp in self.components],
            "connections": [conn.copy() for conn in self.connections],
            "deployment": self.deployment.copy(),
            "patterns": self.patterns.copy()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Architecture":
        """
        Deserialize architecture from dictionary.

        Args:
            data: Dictionary containing architecture fields

        Returns:
            Architecture instance

        Example:
            >>> data = {
            ...     "components": [{"name": "service"}],
            ...     "connections": [],
            ...     "deployment": {},
            ...     "patterns": ["microservices"]
            ... }
            >>> arch = Architecture.from_dict(data)
            >>> arch.patterns
            ['microservices']
        """
        return cls(
            components=data.get("components", []),
            connections=data.get("connections", []),
            deployment=data.get("deployment", {}),
            patterns=data.get("patterns", [])
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = []
        if self.components:
            parts.append(f"{len(self.components)} component(s)")
        if self.connections:
            parts.append(f"{len(self.connections)} connection(s)")
        if self.patterns:
            parts.append(f"{len(self.patterns)} pattern(s)")

        if not parts:
            return "Architecture(empty)"

        return f"Architecture({', '.join(parts)})"
