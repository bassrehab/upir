"""
Pattern extraction using clustering.

This module implements pattern discovery from collections of UPIR instances
using unsupervised clustering to identify common architectural structures.

Implementation based on:
- scikit-learn clustering: https://scikit-learn.org/stable/modules/clustering.html
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Pattern recognition techniques

Author: Subhadip Mitra
License: Apache 2.0
"""

import logging
from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from upir.core.upir import UPIR
from upir.patterns.pattern import Pattern

logger = logging.getLogger(__name__)


class PatternExtractor:
    """
    Extract architectural patterns from UPIR instances using clustering.

    Discovers common architectural patterns by:
    1. Extracting features from each UPIR architecture
    2. Clustering similar architectures using KMeans
    3. Analyzing each cluster to identify common structure
    4. Creating pattern templates from clusters

    The extractor can identify patterns like "streaming ETL", "API gateway",
    "batch processing", etc., based on structural similarities.

    Attributes:
        n_clusters: Number of clusters for KMeans (default 10)
        scaler: StandardScaler for feature normalization
        kmeans: KMeans clustering model
        feature_dim: Dimension of feature vectors

    Example:
        >>> extractor = PatternExtractor(n_clusters=5)
        >>> upirs = [...]  # List of UPIR instances
        >>> patterns = extractor.discover_patterns(upirs)
        >>> for pattern in patterns:
        ...     print(f"{pattern.name}: {len(pattern.instances)} instances")

    References:
    - KMeans: Partitional clustering algorithm
    - TD Commons: Pattern extraction for architecture optimization
    - Feature engineering for architectural analysis
    """

    def __init__(self, n_clusters: int = 10, feature_dim: int = 32):
        """
        Initialize pattern extractor.

        Args:
            n_clusters: Number of patterns to discover
            feature_dim: Dimension of feature vectors (default 32)
        """
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        logger.info(
            f"Initialized PatternExtractor: n_clusters={n_clusters}, "
            f"feature_dim={feature_dim}"
        )

    def extract_features(self, upir: UPIR) -> np.ndarray:
        """
        Extract feature vector from UPIR architecture.

        Extracts architectural features and creates a fixed-size normalized
        feature vector suitable for clustering. Features include:
        - Component count
        - Connection density (connections / components)
        - Deployment type (one-hot encoding)
        - Component type distribution
        - Constraint profile (latency, throughput requirements)

        Args:
            upir: UPIR instance to extract features from

        Returns:
            Feature vector (feature_dim,) normalized to [0, 1]

        Example:
            >>> extractor = PatternExtractor()
            >>> upir = UPIR(id="test", name="Test", description="Test")
            >>> features = extractor.extract_features(upir)
            >>> features.shape
            (32,)
        """
        features = []

        if upir.architecture is None:
            # No architecture - return zero vector
            return np.zeros(self.feature_dim)

        arch = upir.architecture

        # Basic structural features
        num_components = len(arch.components)
        num_connections = len(arch.connections)

        features.append(num_components)
        features.append(num_connections)

        # Connection density
        connection_density = num_connections / max(num_components, 1)
        features.append(connection_density)

        # Component type distribution (one-hot for common types)
        component_types = Counter()
        for comp in arch.components:
            if isinstance(comp, dict):
                comp_type = comp.get("type", "unknown")
            else:
                comp_type = "unknown"
            component_types[comp_type] += 1

        # Top component types (streaming, batch, api, database, cache)
        common_types = ["streaming", "batch", "api", "database", "cache"]
        for comp_type in common_types:
            count = sum(component_types[k] for k in component_types if comp_type in k.lower())
            features.append(count / max(num_components, 1))

        # Deployment pattern (based on deployment config)
        deployment_types = ["single_region", "multi_region", "distributed", "serverless"]
        deployment = arch.deployment if hasattr(arch, 'deployment') else {}
        if isinstance(deployment, dict):
            deployment_type = deployment.get("type", "unknown")
        else:
            deployment_type = "unknown"

        for dep_type in deployment_types:
            features.append(1.0 if dep_type in deployment_type.lower() else 0.0)

        # Constraint profile from specification
        if upir.specification:
            spec = upir.specification

            # Latency constraints
            has_latency = any(
                prop.time_bound is not None
                for prop in spec.properties + spec.invariants
            )
            features.append(1.0 if has_latency else 0.0)

            # Min latency requirement (normalized)
            min_latency = min(
                (prop.time_bound for prop in spec.properties + spec.invariants
                 if prop.time_bound is not None),
                default=0
            )
            features.append(min_latency / 10000.0)  # Normalize to [0, 1]

            # Throughput requirements (heuristic from predicates)
            has_throughput = any(
                "throughput" in prop.predicate.lower() or "qps" in prop.predicate.lower()
                for prop in spec.properties + spec.invariants
            )
            features.append(1.0 if has_throughput else 0.0)

            # Number of constraints
            num_constraints = len(spec.properties) + len(spec.invariants)
            features.append(num_constraints / 10.0)  # Normalize
        else:
            # No specification - add zeros
            features.extend([0.0] * 4)

        # Pad or truncate to fixed size
        features = np.array(features)
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)), mode='constant')
        else:
            features = features[:self.feature_dim]

        # Clip to [0, 1] for numerical stability
        features = np.clip(features, 0.0, 1.0)

        return features.astype(np.float32)

    def cluster_architectures(self, upirs: List[UPIR]) -> Dict[int, List[UPIR]]:
        """
        Cluster similar architectures using KMeans.

        Groups UPIRs with similar architectural features into clusters.
        Each cluster represents a potential pattern.

        Args:
            upirs: List of UPIR instances to cluster

        Returns:
            Dictionary mapping cluster_id -> list of UPIRs in that cluster

        Example:
            >>> extractor = PatternExtractor(n_clusters=3)
            >>> upirs = [...]  # List of UPIRs
            >>> clusters = extractor.cluster_architectures(upirs)
            >>> for cluster_id, cluster_upirs in clusters.items():
            ...     print(f"Cluster {cluster_id}: {len(cluster_upirs)} UPIRs")

        References:
        - KMeans: Minimizes within-cluster sum of squares
        - Feature normalization improves clustering quality
        """
        if not upirs:
            logger.warning("Empty UPIR list provided for clustering")
            return {}

        if len(upirs) < self.n_clusters:
            logger.warning(
                f"Only {len(upirs)} UPIRs but {self.n_clusters} clusters requested. "
                f"Using {len(upirs)} clusters instead."
            )
            self.n_clusters = len(upirs)
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)

        # Extract features from all UPIRs
        feature_matrix = np.array([self.extract_features(upir) for upir in upirs])

        # Normalize features
        feature_matrix_normalized = self.scaler.fit_transform(feature_matrix)

        # Cluster
        labels = self.kmeans.fit_predict(feature_matrix_normalized)

        # Group UPIRs by cluster
        clusters = {}
        for cluster_id in range(self.n_clusters):
            clusters[cluster_id] = []

        for upir, label in zip(upirs, labels):
            clusters[label].append(upir)

        logger.info(
            f"Clustered {len(upirs)} UPIRs into {self.n_clusters} clusters. "
            f"Sizes: {[len(clusters[i]) for i in range(self.n_clusters)]}"
        )

        return clusters

    def extract_pattern(self, cluster: List[UPIR], cluster_id: int) -> Pattern:
        """
        Extract common pattern from a cluster of similar architectures.

        Analyzes UPIRs in the cluster to identify common structure and
        creates a reusable pattern template.

        Args:
            cluster: List of UPIRs in the cluster
            cluster_id: Cluster identifier

        Returns:
            Pattern extracted from the cluster

        Example:
            >>> extractor = PatternExtractor()
            >>> cluster_upirs = [...]  # UPIRs from same cluster
            >>> pattern = extractor.extract_pattern(cluster_upirs, cluster_id=0)
            >>> print(pattern.name)

        References:
        - Mode for categorical features
        - Average for numerical features
        - Template parameterization
        """
        if not cluster:
            # Empty cluster - return empty pattern
            return Pattern(
                id=f"pattern-{cluster_id}",
                name=f"Pattern {cluster_id}",
                description="Empty pattern",
                template={}
            )

        # Collect statistics from cluster
        component_types = Counter()
        connection_counts = []
        component_counts = []
        has_specifications = 0
        performance_metrics = []

        for upir in cluster:
            if upir.architecture:
                arch = upir.architecture

                # Count components by type
                for comp in arch.components:
                    if isinstance(comp, dict):
                        comp_type = comp.get("type", "unknown")
                        component_types[comp_type] += 1

                component_counts.append(len(arch.components))
                connection_counts.append(len(arch.connections))

            if upir.specification:
                has_specifications += 1

        # Identify most common component types
        top_types = component_types.most_common(5)

        # Create template structure
        template_components = []
        for comp_type, count in top_types:
            template_components.append({
                "type": comp_type,
                "count": count / len(cluster),  # Average count per UPIR
                "properties": {}
            })

        # Average metrics
        avg_components = np.mean(component_counts) if component_counts else 0
        avg_connections = np.mean(connection_counts) if connection_counts else 0

        # Compute pattern centroid for matching
        feature_matrix = np.array([self.extract_features(upir) for upir in cluster])
        centroid = np.mean(feature_matrix, axis=0).tolist()

        # Create pattern name (heuristic from top component type)
        if top_types:
            top_type = top_types[0][0]
            pattern_name = f"{top_type.replace('_', ' ').title()} Pattern"
        else:
            pattern_name = f"Pattern {cluster_id}"

        # Create description
        description = (
            f"Architectural pattern with {avg_components:.1f} components "
            f"and {avg_connections:.1f} connections on average. "
            f"Common component types: {', '.join(t for t, _ in top_types[:3])}. "
            f"Specification coverage: {has_specifications}/{len(cluster)}."
        )

        # Create pattern
        pattern = Pattern(
            id=f"pattern-{cluster_id}",
            name=pattern_name,
            description=description,
            template={
                "components": template_components,
                "parameters": {
                    "avg_component_count": avg_components,
                    "avg_connection_count": avg_connections,
                },
                "centroid": centroid,
            },
            instances=[upir.id for upir in cluster],
            success_rate=has_specifications / len(cluster) if cluster else 0.0,
            average_performance={}
        )

        logger.debug(
            f"Extracted pattern: {pattern.name} from {len(cluster)} UPIRs"
        )

        return pattern

    def discover_patterns(self, upirs: List[UPIR]) -> List[Pattern]:
        """
        Discover architectural patterns from UPIR instances.

        Main entry point for pattern extraction. Performs clustering and
        pattern extraction in one pipeline.

        Args:
            upirs: List of UPIR instances to analyze

        Returns:
            List of discovered patterns

        Example:
            >>> extractor = PatternExtractor(n_clusters=5)
            >>> upirs = [...]  # Collection of UPIR instances
            >>> patterns = extractor.discover_patterns(upirs)
            >>> print(f"Discovered {len(patterns)} patterns")
            >>> for pattern in patterns:
            ...     print(f"  - {pattern.name}: {len(pattern.instances)} instances")

        References:
        - TD Commons: Pattern-based architecture optimization
        - Unsupervised learning for pattern discovery
        """
        if not upirs:
            logger.warning("No UPIRs provided for pattern discovery")
            return []

        logger.info(f"Starting pattern discovery for {len(upirs)} UPIRs")

        # Cluster architectures
        clusters = self.cluster_architectures(upirs)

        # Extract pattern from each cluster
        patterns = []
        for cluster_id, cluster_upirs in clusters.items():
            if cluster_upirs:  # Skip empty clusters
                pattern = self.extract_pattern(cluster_upirs, cluster_id)
                patterns.append(pattern)

        # Sort patterns by instance count (descending)
        patterns.sort(key=lambda p: len(p.instances), reverse=True)

        logger.info(
            f"Discovered {len(patterns)} patterns. "
            f"Top pattern: {patterns[0].name} with {len(patterns[0].instances)} instances"
            if patterns else "No patterns discovered"
        )

        return patterns

    def classify_upir(self, upir: UPIR) -> int:
        """
        Classify a UPIR into an existing cluster/pattern.

        Uses the trained KMeans model to predict which cluster a new UPIR
        belongs to.

        Args:
            upir: UPIR to classify

        Returns:
            Cluster ID (0 to n_clusters-1)

        Example:
            >>> extractor = PatternExtractor()
            >>> extractor.discover_patterns(training_upirs)
            >>> new_upir = UPIR(...)
            >>> cluster_id = extractor.classify_upir(new_upir)
        """
        features = self.extract_features(upir)
        features_normalized = self.scaler.transform(features.reshape(1, -1))
        cluster_id = self.kmeans.predict(features_normalized)[0]
        return cluster_id

    def __str__(self) -> str:
        """String representation."""
        return f"PatternExtractor(n_clusters={self.n_clusters})"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"PatternExtractor(n_clusters={self.n_clusters}, "
            f"feature_dim={self.feature_dim})"
        )
