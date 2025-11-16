"""
Pattern library for storing and retrieving architectural patterns.

Provides a centralized repository of reusable architectural patterns with
search, matching, and persistence capabilities.

Implementation based on:
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Distributed systems patterns literature
- Martin Fowler patterns: https://martinfowler.com/

Author: Subhadip Mitra
License: Apache 2.0
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from upir.core.upir import UPIR
from upir.patterns.extractor import PatternExtractor
from upir.patterns.pattern import Pattern

logger = logging.getLogger(__name__)


class PatternLibrary:
    """
    Centralized library for architectural patterns.

    Manages a collection of reusable architectural patterns with capabilities
    for storage, retrieval, search, and similarity-based matching.

    Features:
    - Pattern storage and retrieval
    - Similarity-based architecture matching
    - Search by component types and constraints
    - Success rate tracking with Bayesian updates
    - JSON persistence
    - Built-in common distributed system patterns

    Attributes:
        storage_path: Path to JSON storage file
        patterns: Dictionary mapping pattern_id -> Pattern
        extractor: PatternExtractor for feature extraction

    Example:
        >>> library = PatternLibrary("patterns.json")
        >>> library.load()  # Load existing patterns
        >>> matches = library.match_architecture(upir, threshold=0.8)
        >>> for pattern, score in matches:
        ...     print(f"{pattern.name}: {score:.2f}")

    References:
    - TD Commons: Pattern-based optimization
    - Cosine similarity for vector matching
    - Bayesian updates for success rate tracking
    """

    def __init__(self, storage_path: str = "patterns.json"):
        """
        Initialize pattern library.

        Args:
            storage_path: Path to JSON file for pattern storage
        """
        self.storage_path = Path(storage_path)
        self.patterns: Dict[str, Pattern] = {}
        self.extractor = PatternExtractor(feature_dim=32)

        # Initialize with built-in patterns
        self._initialize_builtin_patterns()

        logger.info(f"Initialized PatternLibrary: storage={storage_path}")

    def add_pattern(self, pattern: Pattern) -> str:
        """
        Add pattern to library.

        Args:
            pattern: Pattern to add

        Returns:
            Pattern ID

        Example:
            >>> library = PatternLibrary()
            >>> pattern = Pattern(id="custom-1", name="Custom", ...)
            >>> pattern_id = library.add_pattern(pattern)
        """
        self.patterns[pattern.id] = pattern
        logger.info(f"Added pattern: {pattern.id} ({pattern.name})")
        return pattern.id

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """
        Get pattern by ID.

        Args:
            pattern_id: Pattern identifier

        Returns:
            Pattern if found, None otherwise

        Example:
            >>> library = PatternLibrary()
            >>> pattern = library.get_pattern("streaming-etl")
        """
        return self.patterns.get(pattern_id)

    def search_patterns(self, query: Dict[str, Any]) -> List[Pattern]:
        """
        Search patterns by component types, constraints, etc.

        Supports queries on:
        - component_types: List of required component types
        - min_success_rate: Minimum success rate threshold
        - has_specification: Whether pattern includes specifications
        - name_contains: Substring match on pattern name

        Args:
            query: Search criteria dictionary

        Returns:
            List of matching patterns

        Example:
            >>> library = PatternLibrary()
            >>> results = library.search_patterns({
            ...     "component_types": ["streaming_processor"],
            ...     "min_success_rate": 0.8
            ... })
        """
        results = []

        for pattern in self.patterns.values():
            # Check component types
            if "component_types" in query:
                required_types = set(query["component_types"])
                pattern_types = set()
                for comp in pattern.template.get("components", []):
                    pattern_types.add(comp.get("type", ""))

                # Must have all required types
                if not required_types.issubset(pattern_types):
                    continue

            # Check minimum success rate
            if "min_success_rate" in query:
                if pattern.success_rate < query["min_success_rate"]:
                    continue

            # Check if has specification
            if "has_specification" in query:
                has_spec = pattern.success_rate > 0.0  # Heuristic
                if query["has_specification"] != has_spec:
                    continue

            # Check name contains substring
            if "name_contains" in query:
                if query["name_contains"].lower() not in pattern.name.lower():
                    continue

            results.append(pattern)

        logger.debug(f"Search found {len(results)} patterns matching {query}")
        return results

    def match_architecture(
        self, upir: UPIR, threshold: float = 0.8
    ) -> List[Tuple[Pattern, float]]:
        """
        Find patterns matching UPIR architecture.

        Computes similarity between UPIR and all patterns in library,
        returning patterns above threshold sorted by similarity score.

        Similarity scoring:
        - Base: Cosine similarity of feature vectors
        - Bonus: +0.1 for exact component type matches
        - Weight: Multiply by (0.5 + 0.5 * success_rate)

        Args:
            upir: UPIR to match against patterns
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of (Pattern, similarity_score) tuples, sorted by score descending

        Example:
            >>> library = PatternLibrary()
            >>> matches = library.match_architecture(upir, threshold=0.7)
            >>> best_match, score = matches[0]
            >>> print(f"Best match: {best_match.name} ({score:.2%})")
        """
        # Extract features from UPIR
        upir_features = self.extractor.extract_features(upir)

        matches = []

        for pattern in self.patterns.values():
            # Get pattern centroid from template
            centroid = pattern.template.get("centroid")
            if not centroid:
                continue

            # Compute cosine similarity
            centroid_array = np.array(centroid).reshape(1, -1)
            upir_array = upir_features.reshape(1, -1)
            similarity = cosine_similarity(upir_array, centroid_array)[0][0]

            # Bonus for exact component type matches
            bonus = self._compute_component_match_bonus(upir, pattern)
            similarity += bonus

            # Weight by success rate (50% base + 50% success rate)
            weight = 0.5 + 0.5 * pattern.success_rate
            weighted_similarity = similarity * weight

            # Clip to [0, 1]
            weighted_similarity = np.clip(weighted_similarity, 0.0, 1.0)

            # Only include if above threshold
            if weighted_similarity >= threshold:
                matches.append((pattern, float(weighted_similarity)))

        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Matched {len(matches)} patterns for UPIR {upir.id} "
            f"(threshold={threshold})"
        )

        return matches

    def _compute_component_match_bonus(self, upir: UPIR, pattern: Pattern) -> float:
        """
        Compute bonus for exact component type matches.

        Args:
            upir: UPIR to match
            pattern: Pattern to match against

        Returns:
            Bonus score (0.0 to 0.1)
        """
        if not upir.architecture:
            return 0.0

        # Get component types from UPIR
        upir_types = set()
        for comp in upir.architecture.components:
            if isinstance(comp, dict):
                upir_types.add(comp.get("type", "unknown"))

        # Get component types from pattern
        pattern_types = set()
        for comp in pattern.template.get("components", []):
            pattern_types.add(comp.get("type", "unknown"))

        # Compute overlap
        if not pattern_types:
            return 0.0

        overlap = len(upir_types & pattern_types)
        total = len(pattern_types)

        # Up to +0.1 bonus
        return 0.1 * (overlap / total)

    def update_success_rate(self, pattern_id: str, success: bool):
        """
        Update pattern success statistics.

        Uses Bayesian updating with Beta distribution:
        - Prior: Inferred from current success_rate (as pseudo-observations)
        - Update: Add 1 to α (successes) or β (failures)
        - Posterior mean: α / (α + β)

        For patterns with no instances, treats current success_rate as
        coming from 10 pseudo-observations to provide informative prior.

        Args:
            pattern_id: Pattern to update
            success: Whether pattern was successfully applied

        Example:
            >>> library = PatternLibrary()
            >>> library.update_success_rate("streaming-etl", success=True)
        """
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            logger.warning(f"Pattern {pattern_id} not found for success update")
            return

        # Bayesian update with Beta distribution
        # Current success_rate is posterior mean: α / (α + β)

        # Infer current α, β from success_rate and instance count
        n_instances = len(pattern.instances)
        if n_instances == 0:
            # No instances yet - use success_rate as informative prior
            # Treat as if we had 10 pseudo-observations
            pseudo_count = 10.0
            alpha = pattern.success_rate * pseudo_count
            beta = (1.0 - pattern.success_rate) * pseudo_count
        else:
            # Infer from current success rate
            # success_rate = α / (α + β)
            # Assume total = α + β = n_instances + 2 (prior)
            total = n_instances + 2
            alpha = pattern.success_rate * total
            beta = total - alpha

        # Update with new observation
        if success:
            alpha += 1
        else:
            beta += 1

        # Compute new success rate (posterior mean)
        new_success_rate = alpha / (alpha + beta)

        pattern.success_rate = new_success_rate

        logger.info(
            f"Updated success rate for {pattern_id}: "
            f"{pattern.success_rate:.3f} (success={success})"
        )

    def save(self):
        """
        Save patterns to JSON file.

        Example:
            >>> library = PatternLibrary("patterns.json")
            >>> library.save()
        """
        data = {
            "patterns": [pattern.to_dict() for pattern in self.patterns.values()]
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.patterns)} patterns to {self.storage_path}")

    def load(self):
        """
        Load patterns from JSON file.

        If file doesn't exist, keeps built-in patterns only.

        Example:
            >>> library = PatternLibrary("patterns.json")
            >>> library.load()
        """
        if not self.storage_path.exists():
            logger.info(f"Storage file {self.storage_path} not found, using built-in patterns")
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            self.patterns = {}
            for pattern_data in data.get("patterns", []):
                pattern = Pattern.from_dict(pattern_data)
                self.patterns[pattern.id] = pattern

            logger.info(f"Loaded {len(self.patterns)} patterns from {self.storage_path}")

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            # Keep built-in patterns on error
            self._initialize_builtin_patterns()

    def _initialize_builtin_patterns(self):
        """Initialize library with common distributed system patterns."""

        # 1. Streaming ETL Pipeline
        self.patterns["streaming-etl"] = Pattern(
            id="streaming-etl",
            name="Streaming ETL Pipeline",
            description=(
                "Real-time data pipeline with message queue, streaming processor, "
                "and persistent storage. Typical latency: <100ms, high throughput."
            ),
            template={
                "components": [
                    {"type": "pubsub_source", "count": 1.0, "properties": {}},
                    {"type": "streaming_processor", "count": 1.0, "properties": {}},
                    {"type": "database", "count": 1.0, "properties": {}},
                ],
                "parameters": {
                    "avg_component_count": 3.0,
                    "avg_connection_count": 2.0,
                },
                "centroid": self._compute_centroid(
                    ["pubsub_source", "streaming_processor", "database"],
                    connections=2,
                    has_latency=True,
                    latency_ms=100,
                ),
            },
            success_rate=0.85,
        )

        # 2. Batch Processing
        self.patterns["batch-processing"] = Pattern(
            id="batch-processing",
            name="Batch Processing Pipeline",
            description=(
                "Scheduled batch jobs processing large datasets. "
                "High throughput, relaxed latency requirements."
            ),
            template={
                "components": [
                    {"type": "bigquery_source", "count": 1.0, "properties": {}},
                    {"type": "batch_processor", "count": 1.0, "properties": {}},
                    {"type": "database", "count": 1.0, "properties": {}},
                ],
                "parameters": {
                    "avg_component_count": 3.0,
                    "avg_connection_count": 2.0,
                },
                "centroid": self._compute_centroid(
                    ["bigquery_source", "batch_processor", "database"],
                    connections=2,
                    has_latency=False,
                ),
            },
            success_rate=0.90,
        )

        # 3. Request-Response API
        self.patterns["api-gateway"] = Pattern(
            id="api-gateway",
            name="Request-Response API",
            description=(
                "Synchronous API with gateway, service layer, and database. "
                "Low latency (<50ms), moderate throughput."
            ),
            template={
                "components": [
                    {"type": "api_gateway", "count": 1.0, "properties": {}},
                    {"type": "api_server", "count": 1.0, "properties": {}},
                    {"type": "database", "count": 1.0, "properties": {}},
                ],
                "parameters": {
                    "avg_component_count": 3.0,
                    "avg_connection_count": 2.0,
                },
                "centroid": self._compute_centroid(
                    ["api_gateway", "api_server", "database"],
                    connections=2,
                    has_latency=True,
                    latency_ms=50,
                ),
            },
            success_rate=0.88,
        )

        # 4. Event-Driven Microservices
        self.patterns["event-driven"] = Pattern(
            id="event-driven",
            name="Event-Driven Microservices",
            description=(
                "Asynchronous microservices communicating via events. "
                "Decoupled, scalable, eventually consistent."
            ),
            template={
                "components": [
                    {"type": "pubsub", "count": 1.0, "properties": {}},
                    {"type": "streaming_processor", "count": 3.0, "properties": {}},
                    {"type": "database", "count": 2.0, "properties": {}},
                ],
                "parameters": {
                    "avg_component_count": 6.0,
                    "avg_connection_count": 5.0,
                },
                "centroid": self._compute_centroid(
                    ["pubsub", "streaming_processor", "streaming_processor",
                     "streaming_processor", "database", "database"],
                    connections=5,
                    has_latency=True,
                    latency_ms=200,
                ),
            },
            success_rate=0.80,
        )

        # 5. Lambda Architecture
        self.patterns["lambda-architecture"] = Pattern(
            id="lambda-architecture",
            name="Lambda Architecture",
            description=(
                "Hybrid batch and streaming layers with serving layer. "
                "Combines batch accuracy with streaming speed."
            ),
            template={
                "components": [
                    {"type": "pubsub_source", "count": 1.0, "properties": {}},
                    {"type": "streaming_processor", "count": 1.0, "properties": {}},
                    {"type": "batch_processor", "count": 1.0, "properties": {}},
                    {"type": "database", "count": 2.0, "properties": {}},
                ],
                "parameters": {
                    "avg_component_count": 5.0,
                    "avg_connection_count": 4.0,
                },
                "centroid": self._compute_centroid(
                    ["pubsub_source", "streaming_processor", "batch_processor",
                     "database", "database"],
                    connections=4,
                    has_latency=True,
                    latency_ms=150,
                ),
            },
            success_rate=0.75,
        )

        # 6. Kappa Architecture
        self.patterns["kappa-architecture"] = Pattern(
            id="kappa-architecture",
            name="Kappa Architecture",
            description=(
                "Streaming-only architecture without batch layer. "
                "Simpler than Lambda, reprocessing via stream replay."
            ),
            template={
                "components": [
                    {"type": "pubsub_source", "count": 1.0, "properties": {}},
                    {"type": "streaming_processor", "count": 2.0, "properties": {}},
                    {"type": "database", "count": 1.0, "properties": {}},
                ],
                "parameters": {
                    "avg_component_count": 4.0,
                    "avg_connection_count": 3.0,
                },
                "centroid": self._compute_centroid(
                    ["pubsub_source", "streaming_processor", "streaming_processor", "database"],
                    connections=3,
                    has_latency=True,
                    latency_ms=100,
                ),
            },
            success_rate=0.82,
        )

        # 7. CQRS (Command Query Responsibility Segregation)
        self.patterns["cqrs"] = Pattern(
            id="cqrs",
            name="CQRS Pattern",
            description=(
                "Separate read and write models with different databases. "
                "Optimized queries, eventual consistency."
            ),
            template={
                "components": [
                    {"type": "api_server", "count": 1.0, "properties": {}},
                    {"type": "database", "count": 2.0, "properties": {}},
                    {"type": "cache", "count": 1.0, "properties": {}},
                ],
                "parameters": {
                    "avg_component_count": 4.0,
                    "avg_connection_count": 3.0,
                },
                "centroid": self._compute_centroid(
                    ["api_server", "database", "database", "cache"],
                    connections=3,
                    has_latency=True,
                    latency_ms=30,
                ),
            },
            success_rate=0.83,
        )

        # 8. Event Sourcing
        self.patterns["event-sourcing"] = Pattern(
            id="event-sourcing",
            name="Event Sourcing",
            description=(
                "Store state changes as event log. "
                "Full audit trail, temporal queries, replay capability."
            ),
            template={
                "components": [
                    {"type": "api_server", "count": 1.0, "properties": {}},
                    {"type": "pubsub", "count": 1.0, "properties": {}},
                    {"type": "database", "count": 1.0, "properties": {}},
                    {"type": "cache", "count": 1.0, "properties": {}},
                ],
                "parameters": {
                    "avg_component_count": 4.0,
                    "avg_connection_count": 3.0,
                },
                "centroid": self._compute_centroid(
                    ["api_server", "pubsub", "database", "cache"],
                    connections=3,
                    has_latency=True,
                    latency_ms=50,
                ),
            },
            success_rate=0.78,
        )

        # 9. Pub/Sub Fanout
        self.patterns["pubsub-fanout"] = Pattern(
            id="pubsub-fanout",
            name="Pub/Sub Fanout",
            description=(
                "Single publisher with multiple subscribers. "
                "Decoupled communication, parallel processing."
            ),
            template={
                "components": [
                    {"type": "pubsub", "count": 1.0, "properties": {}},
                    {"type": "streaming_processor", "count": 4.0, "properties": {}},
                    {"type": "database", "count": 2.0, "properties": {}},
                ],
                "parameters": {
                    "avg_component_count": 7.0,
                    "avg_connection_count": 6.0,
                },
                "centroid": self._compute_centroid(
                    ["pubsub", "streaming_processor", "streaming_processor",
                     "streaming_processor", "streaming_processor", "database", "database"],
                    connections=6,
                    has_latency=True,
                    latency_ms=100,
                ),
            },
            success_rate=0.86,
        )

        # 10. MapReduce
        self.patterns["mapreduce"] = Pattern(
            id="mapreduce",
            name="MapReduce",
            description=(
                "Parallel batch processing with map and reduce phases. "
                "Large-scale data processing, high throughput."
            ),
            template={
                "components": [
                    {"type": "bigquery_source", "count": 1.0, "properties": {}},
                    {"type": "batch_processor", "count": 3.0, "properties": {}},
                    {"type": "database", "count": 1.0, "properties": {}},
                ],
                "parameters": {
                    "avg_component_count": 5.0,
                    "avg_connection_count": 4.0,
                },
                "centroid": self._compute_centroid(
                    ["bigquery_source", "batch_processor", "batch_processor",
                     "batch_processor", "database"],
                    connections=4,
                    has_latency=False,
                ),
            },
            success_rate=0.87,
        )

        logger.info(f"Initialized {len(self.patterns)} built-in patterns")

    def _compute_centroid(
        self,
        component_types: List[str],
        connections: int,
        has_latency: bool = False,
        latency_ms: float = 0.0,
    ) -> List[float]:
        """
        Compute feature centroid for pattern template.

        Creates a synthetic UPIR from pattern parameters and extracts features.

        Args:
            component_types: List of component type names
            connections: Number of connections
            has_latency: Whether pattern has latency constraints
            latency_ms: Latency bound in milliseconds

        Returns:
            32-dimensional feature vector
        """
        from upir.core.architecture import Architecture
        from upir.core.specification import FormalSpecification
        from upir.core.temporal import TemporalOperator, TemporalProperty

        # Create synthetic components
        components = [
            {"id": f"c{i}", "name": f"C{i}", "type": comp_type}
            for i, comp_type in enumerate(component_types)
        ]

        # Create synthetic connections
        connections_list = []
        for i in range(min(connections, len(components) - 1)):
            connections_list.append({"from": f"c{i}", "to": f"c{i+1}"})

        arch = Architecture(components=components, connections=connections_list)

        # Create specification if needed
        spec = None
        if has_latency:
            spec = FormalSpecification(
                properties=[
                    TemporalProperty(
                        operator=TemporalOperator.WITHIN,
                        predicate="process",
                        time_bound=latency_ms,
                    )
                ]
            )

        # Create UPIR and extract features
        upir = UPIR(
            id="synthetic",
            name="Synthetic",
            description="Synthetic for centroid",
            architecture=arch,
            specification=spec,
        )

        features = self.extractor.extract_features(upir)
        return features.tolist()

    def __len__(self) -> int:
        """Return number of patterns in library."""
        return len(self.patterns)

    def __contains__(self, pattern_id: str) -> bool:
        """Check if pattern exists in library."""
        return pattern_id in self.patterns

    def __str__(self) -> str:
        """String representation."""
        return f"PatternLibrary({len(self.patterns)} patterns)"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"PatternLibrary(storage_path='{self.storage_path}', n_patterns={len(self.patterns)})"
