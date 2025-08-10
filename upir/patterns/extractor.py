"""
Pattern Extraction System for UPIR

This module discovers reusable architectural patterns from multiple UPIR
instances using machine learning techniques. It identifies common structures,
abstracts them into patterns, and enables pattern reuse across projects.

The key insight: Many distributed systems share common architectural patterns.
By extracting and parameterizing these patterns, we can:
1. Speed up architecture design
2. Apply proven solutions to new problems
3. Learn which patterns work best for different scenarios

Author: subhadipmitra@google.com
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import json
import hashlib
import logging
from collections import defaultdict

# For clustering
try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not installed. Pattern extraction will be limited.")

from ..core.models import UPIR, Architecture, FormalSpecification

logger = logging.getLogger(__name__)


@dataclass
class PatternFeatures:
    """
    Features extracted from an architecture for pattern discovery.
    
    We extract both structural and behavioral features to capture
    the essence of architectural patterns.
    """
    # Structural features
    num_components: int
    num_connections: int
    component_types: Dict[str, int]  # Count of each component type
    connection_density: float  # Connections per component
    max_fan_out: int  # Maximum outgoing connections from a component
    max_fan_in: int  # Maximum incoming connections to a component
    
    # Topological features
    has_cycles: bool
    depth: int  # Longest path in the architecture
    breadth: int  # Maximum width at any level
    clustering_coefficient: float  # How clustered the components are
    
    # Pattern indicators
    uses_caching: bool
    uses_queuing: bool
    uses_replication: bool
    uses_sharding: bool
    uses_load_balancing: bool
    
    # Performance characteristics (if available)
    avg_latency: Optional[float] = None
    avg_throughput: Optional[float] = None
    success_rate: Optional[float] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numerical vector for clustering."""
        vector = []
        
        # Numerical features
        vector.append(float(self.num_components))
        vector.append(float(self.num_connections))
        vector.append(self.connection_density)
        vector.append(float(self.max_fan_out))
        vector.append(float(self.max_fan_in))
        vector.append(float(self.depth))
        vector.append(float(self.breadth))
        vector.append(self.clustering_coefficient)
        
        # Component type distribution
        for comp_type in ["service", "storage", "queue", "cache", "processor", "gateway"]:
            vector.append(float(self.component_types.get(comp_type, 0)))
        
        # Boolean features as 0/1
        vector.append(1.0 if self.has_cycles else 0.0)
        vector.append(1.0 if self.uses_caching else 0.0)
        vector.append(1.0 if self.uses_queuing else 0.0)
        vector.append(1.0 if self.uses_replication else 0.0)
        vector.append(1.0 if self.uses_sharding else 0.0)
        vector.append(1.0 if self.uses_load_balancing else 0.0)
        
        # Performance features (use defaults if not available)
        vector.append(self.avg_latency if self.avg_latency else 100.0)
        vector.append(self.avg_throughput if self.avg_throughput else 1000.0)
        vector.append(self.success_rate if self.success_rate else 0.95)
        
        return np.array(vector, dtype=np.float32)


@dataclass
class ArchitecturalPattern:
    """
    A discovered architectural pattern.
    
    Patterns are abstracted, parameterized templates that can be
    instantiated for specific use cases.
    """
    id: str
    name: str
    description: str
    category: str  # "streaming", "batch", "microservices", etc.
    
    # Pattern structure
    template_components: List[Dict[str, Any]]  # Parameterized components
    template_connections: List[Dict[str, Any]]  # How components connect
    required_properties: List[str]  # Properties this pattern guarantees
    
    # Pattern metadata
    instances: List[str] = field(default_factory=list)  # UPIR IDs using this pattern
    success_rate: float = 0.0  # How often this pattern succeeds
    avg_performance: Dict[str, float] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    
    # Pattern parameters
    parameters: Dict[str, Any] = field(default_factory=dict)  # Configurable aspects
    constraints: Dict[str, Any] = field(default_factory=dict)  # Parameter constraints
    
    def instantiate(self, params: Dict[str, Any]) -> Architecture:
        """
        Instantiate the pattern with specific parameters.
        
        This is where patterns become concrete architectures.
        """
        # Validate parameters
        for param, constraint in self.constraints.items():
            if param in params:
                value = params[param]
                if "min" in constraint and value < constraint["min"]:
                    raise ValueError(f"{param} must be >= {constraint['min']}")
                if "max" in constraint and value > constraint["max"]:
                    raise ValueError(f"{param} must be <= {constraint['max']}")
        
        # Apply parameters to template
        components = []
        for template_comp in self.template_components:
            comp = template_comp.copy()
            
            # Replace parameter placeholders
            if "config" in comp:
                for key, value in comp["config"].items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        param_name = value[2:-1]
                        if param_name in params:
                            comp["config"][key] = params[param_name]
            
            components.append(comp)
        
        # Create connections
        connections = []
        for template_conn in self.template_connections:
            connections.append(template_conn.copy())
        
        return Architecture(
            components=components,
            connections=connections,
            deployment=params.get("deployment", {}),
            patterns=[self.name]
        )
    
    def matches(self, architecture: Architecture, threshold: float = 0.8) -> bool:
        """
        Check if an architecture matches this pattern.
        
        We use structural similarity to determine matches.
        """
        # Check component types match
        pattern_types = {c["type"] for c in self.template_components if "type" in c}
        arch_types = {c.get("type") for c in architecture.components}
        
        type_similarity = len(pattern_types & arch_types) / max(len(pattern_types), 1)
        
        if type_similarity < threshold:
            return False
        
        # Check connection patterns match
        pattern_conn_count = len(self.template_connections)
        arch_conn_count = len(architecture.connections)
        
        if pattern_conn_count > 0:
            conn_similarity = min(arch_conn_count, pattern_conn_count) / pattern_conn_count
            if conn_similarity < threshold:
                return False
        
        return True
    
    def update_statistics(self, success: bool, performance: Dict[str, float]) -> None:
        """Update pattern statistics based on usage."""
        # Update success rate with exponential moving average
        alpha = 0.1  # Learning rate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        
        # Update performance metrics
        for metric, value in performance.items():
            if metric not in self.avg_performance:
                self.avg_performance[metric] = value
            else:
                self.avg_performance[metric] = (1 - alpha) * self.avg_performance[metric] + alpha * value
        
        self.last_used = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "template_components": self.template_components,
            "template_connections": self.template_connections,
            "required_properties": self.required_properties,
            "instances": self.instances,
            "success_rate": self.success_rate,
            "avg_performance": self.avg_performance,
            "discovered_at": self.discovered_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "parameters": self.parameters,
            "constraints": self.constraints
        }


class FeatureExtractor:
    """
    Extracts features from UPIR architectures for pattern discovery.
    
    This analyzes the structure and properties of architectures to
    create feature vectors suitable for clustering.
    """
    
    def extract(self, upir: UPIR) -> PatternFeatures:
        """Extract features from a UPIR instance."""
        if not upir.architecture:
            raise ValueError("UPIR must have an architecture")
        
        arch = upir.architecture
        
        # Count component types
        component_types = defaultdict(int)
        for comp in arch.components:
            comp_type = comp.get("type", "unknown")
            component_types[comp_type] += 1
        
        # Analyze connections
        num_components = len(arch.components)
        num_connections = len(arch.connections)
        connection_density = num_connections / max(num_components, 1)
        
        # Calculate fan-in and fan-out
        fan_out = defaultdict(int)
        fan_in = defaultdict(int)
        
        for conn in arch.connections:
            source = conn.get("source")
            target = conn.get("target")
            if source:
                fan_out[source] += 1
            if target:
                fan_in[target] += 1
        
        max_fan_out = max(fan_out.values()) if fan_out else 0
        max_fan_in = max(fan_in.values()) if fan_in else 0
        
        # Check for cycles (simplified)
        has_cycles = self._has_cycles(arch)
        
        # Calculate depth and breadth
        depth, breadth = self._calculate_topology(arch)
        
        # Calculate clustering coefficient
        clustering = self._calculate_clustering(arch)
        
        # Check for specific patterns
        uses_caching = any("cache" in str(comp).lower() for comp in arch.components)
        uses_queuing = any("queue" in str(comp).lower() or "pubsub" in str(comp).lower() 
                          for comp in arch.components)
        uses_replication = any(comp.get("config", {}).get("replicas", 1) > 1 
                             for comp in arch.components)
        uses_sharding = any("shard" in str(comp).lower() for comp in arch.components)
        uses_load_balancing = any("balancer" in str(comp).lower() or "lb" in str(comp).lower()
                                 for comp in arch.components)
        
        # Extract performance if available
        avg_latency = None
        avg_throughput = None
        success_rate = None
        
        if upir.evidence:
            perf_evidence = [e for e in upir.evidence.values() 
                           if e.type in ["benchmark", "production"]]
            if perf_evidence:
                latencies = []
                throughputs = []
                
                for evidence in perf_evidence:
                    if "latency" in evidence.data:
                        latencies.append(evidence.data["latency"])
                    if "throughput" in evidence.data:
                        throughputs.append(evidence.data["throughput"])
                
                if latencies:
                    avg_latency = np.mean(latencies)
                if throughputs:
                    avg_throughput = np.mean(throughputs)
                
                # Use confidence as proxy for success rate
                success_rate = np.mean([e.confidence for e in perf_evidence])
        
        return PatternFeatures(
            num_components=num_components,
            num_connections=num_connections,
            component_types=dict(component_types),
            connection_density=connection_density,
            max_fan_out=max_fan_out,
            max_fan_in=max_fan_in,
            has_cycles=has_cycles,
            depth=depth,
            breadth=breadth,
            clustering_coefficient=clustering,
            uses_caching=uses_caching,
            uses_queuing=uses_queuing,
            uses_replication=uses_replication,
            uses_sharding=uses_sharding,
            uses_load_balancing=uses_load_balancing,
            avg_latency=avg_latency,
            avg_throughput=avg_throughput,
            success_rate=success_rate
        )
    
    def _has_cycles(self, architecture: Architecture) -> bool:
        """Check if architecture has cycles."""
        # Build adjacency list
        graph = defaultdict(list)
        for conn in architecture.connections:
            source = conn.get("source")
            target = conn.get("target")
            if source and target:
                graph[source].append(target)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle_util(node):
                    return True
        
        return False
    
    def _calculate_topology(self, architecture: Architecture) -> Tuple[int, int]:
        """Calculate depth and breadth of architecture."""
        if not architecture.components:
            return 0, 0
        
        # Build graph
        graph = defaultdict(list)
        for conn in architecture.connections:
            source = conn.get("source")
            target = conn.get("target")
            if source and target:
                graph[source].append(target)
        
        # Find roots (components with no incoming edges)
        all_components = {c.get("name", f"comp_{i}") 
                         for i, c in enumerate(architecture.components)}
        targets = {conn.get("target") for conn in architecture.connections}
        roots = all_components - targets
        
        if not roots:
            # If no roots (all have incoming edges), pick arbitrary start
            roots = {list(all_components)[0]} if all_components else set()
        
        # BFS to calculate depth and breadth
        max_depth = 0
        max_breadth = 0
        
        for root in roots:
            visited = set()
            queue = [(root, 0)]
            level_counts = defaultdict(int)
            
            while queue:
                node, depth = queue.pop(0)
                if node in visited:
                    continue
                
                visited.add(node)
                level_counts[depth] += 1
                max_depth = max(max_depth, depth)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
            
            max_breadth = max(max_breadth, max(level_counts.values()) if level_counts else 0)
        
        return max_depth, max_breadth
    
    def _calculate_clustering(self, architecture: Architecture) -> float:
        """
        Calculate clustering coefficient of the architecture.
        
        This measures how tightly connected components are.
        """
        if len(architecture.components) < 3:
            return 0.0
        
        # Build undirected adjacency
        neighbors = defaultdict(set)
        for conn in architecture.connections:
            source = conn.get("source")
            target = conn.get("target")
            if source and target:
                neighbors[source].add(target)
                neighbors[target].add(source)
        
        # Calculate clustering coefficient for each node
        coefficients = []
        
        for node, node_neighbors in neighbors.items():
            k = len(node_neighbors)
            if k < 2:
                continue
            
            # Count edges between neighbors
            edges = 0
            neighbor_list = list(node_neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if neighbor_list[j] in neighbors[neighbor_list[i]]:
                        edges += 1
            
            # Clustering coefficient
            max_edges = k * (k - 1) / 2
            if max_edges > 0:
                coefficients.append(edges / max_edges)
        
        return np.mean(coefficients) if coefficients else 0.0


class PatternClusterer:
    """
    Clusters UPIR instances to discover patterns.
    
    Uses unsupervised learning to group similar architectures
    and extract common patterns.
    """
    
    def __init__(self, method: str = "dbscan"):
        """
        Initialize clusterer.
        
        Args:
            method: Clustering method ("dbscan", "kmeans")
        """
        self.method = method
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.clusterer = None
        self.pca = None
        
    def cluster(self, features_list: List[PatternFeatures], 
               min_cluster_size: int = 2) -> Dict[int, List[int]]:
        """
        Cluster architectures based on features.
        
        Returns mapping from cluster ID to list of indices.
        """
        if not SKLEARN_AVAILABLE:
            # Fallback: simple distance-based clustering
            return self._simple_clustering(features_list)
        
        # Convert features to vectors
        vectors = np.array([f.to_vector() for f in features_list])
        
        # Standardize features
        vectors_scaled = self.scaler.fit_transform(vectors)
        
        # Optionally reduce dimensions for better clustering
        if vectors_scaled.shape[1] > 10:
            self.pca = PCA(n_components=10)
            vectors_scaled = self.pca.fit_transform(vectors_scaled)
        
        # Cluster based on method
        if self.method == "dbscan":
            # DBSCAN for density-based clustering
            self.clusterer = DBSCAN(eps=0.5, min_samples=min_cluster_size)
            labels = self.clusterer.fit_predict(vectors_scaled)
        elif self.method == "kmeans":
            # Determine optimal k using silhouette score
            best_k = 2
            best_score = -1
            
            for k in range(2, min(len(features_list) // 2, 10)):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(vectors_scaled)
                score = silhouette_score(vectors_scaled, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # Final clustering with best k
            self.clusterer = KMeans(n_clusters=best_k, random_state=42)
            labels = self.clusterer.fit_predict(vectors_scaled)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Group by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label >= 0:  # Ignore noise points (-1 in DBSCAN)
                clusters[label].append(idx)
        
        return dict(clusters)
    
    def _simple_clustering(self, features_list: List[PatternFeatures]) -> Dict[int, List[int]]:
        """
        Simple clustering fallback when scikit-learn is not available.
        
        Groups architectures with similar component counts and types.
        """
        clusters = defaultdict(list)
        
        for idx, features in enumerate(features_list):
            # Create simple signature
            signature = (
                features.num_components // 5,  # Group by size
                features.uses_caching,
                features.uses_queuing,
                features.uses_replication
            )
            
            # Hash signature to get cluster ID
            cluster_id = hash(signature) % 100
            clusters[cluster_id].append(idx)
        
        # Filter out single-element clusters
        return {k: v for k, v in clusters.items() if len(v) >= 2}


class PatternAbstractor:
    """
    Abstracts clusters into reusable patterns.
    
    Takes a cluster of similar architectures and creates an
    abstract, parameterized pattern.
    """
    
    def abstract(self, upirs: List[UPIR], name: str = None) -> ArchitecturalPattern:
        """
        Create abstract pattern from similar UPIRs.
        
        This identifies commonalities and variabilities to create
        a parameterized template.
        """
        if not upirs:
            raise ValueError("Need at least one UPIR to abstract")
        
        # Find common component types
        all_component_types = []
        for upir in upirs:
            if upir.architecture:
                types = [c.get("type", "unknown") for c in upir.architecture.components]
                all_component_types.append(types)
        
        # Find most common types
        type_counts = defaultdict(int)
        for types in all_component_types:
            for t in types:
                type_counts[t] += 1
        
        # Components that appear in >50% of instances
        threshold = len(upirs) / 2
        common_types = [t for t, count in type_counts.items() if count >= threshold]
        
        # Create template components
        template_components = []
        for comp_type in common_types:
            template_comp = {
                "type": comp_type,
                "name": f"{comp_type}_${instance_id}",
                "config": {}
            }
            
            # Find common config parameters
            configs = []
            for upir in upirs:
                if upir.architecture:
                    for comp in upir.architecture.components:
                        if comp.get("type") == comp_type and "config" in comp:
                            configs.append(comp["config"])
            
            if configs:
                # Find common keys
                common_keys = set(configs[0].keys())
                for config in configs[1:]:
                    common_keys &= set(config.keys())
                
                # Add parameterized config
                for key in common_keys:
                    values = [c[key] for c in configs]
                    
                    # If values are similar, use average; otherwise parameterize
                    if all(isinstance(v, (int, float)) for v in values):
                        avg_value = np.mean(values)
                        std_value = np.std(values)
                        
                        if std_value / (avg_value + 1e-8) < 0.3:  # Low variance
                            template_comp["config"][key] = avg_value
                        else:  # High variance - parameterize
                            template_comp["config"][key] = f"${{{comp_type}_{key}}}"
            
            template_components.append(template_comp)
        
        # Find common connection patterns
        template_connections = []
        connection_patterns = defaultdict(int)
        
        for upir in upirs:
            if upir.architecture:
                for conn in upir.architecture.connections:
                    source_type = None
                    target_type = None
                    
                    # Find types of source and target
                    for comp in upir.architecture.components:
                        comp_name = comp.get("name")
                        if comp_name == conn.get("source"):
                            source_type = comp.get("type")
                        if comp_name == conn.get("target"):
                            target_type = comp.get("type")
                    
                    if source_type and target_type:
                        pattern = (source_type, target_type)
                        connection_patterns[pattern] += 1
        
        # Add common connection patterns
        for (source_type, target_type), count in connection_patterns.items():
            if count >= threshold:
                template_connections.append({
                    "source": f"{source_type}_${{instance_id}}",
                    "target": f"{target_type}_${{instance_id}}"
                })
        
        # Extract common properties
        required_properties = []
        if upirs[0].specification:
            # Properties that all instances share
            all_props = []
            for upir in upirs:
                if upir.specification:
                    props = [p.predicate for p in upir.specification.invariants]
                    all_props.append(set(props))
            
            if all_props:
                common_props = all_props[0]
                for props in all_props[1:]:
                    common_props &= props
                required_properties = list(common_props)
        
        # Determine category
        category = self._determine_category(upirs)
        
        # Generate pattern ID and name
        pattern_id = hashlib.md5(json.dumps(template_components, sort_keys=True).encode()).hexdigest()[:8]
        
        if not name:
            name = f"{category}_pattern_{pattern_id}"
        
        # Calculate initial statistics
        success_rates = []
        for upir in upirs:
            if upir.evidence:
                confidences = [e.confidence for e in upir.evidence.values()]
                if confidences:
                    success_rates.append(np.mean(confidences))
        
        pattern = ArchitecturalPattern(
            id=pattern_id,
            name=name,
            description=f"Automatically discovered {category} pattern from {len(upirs)} instances",
            category=category,
            template_components=template_components,
            template_connections=template_connections,
            required_properties=required_properties,
            instances=[upir.id for upir in upirs],
            success_rate=np.mean(success_rates) if success_rates else 0.5
        )
        
        # Extract parameters and constraints
        self._extract_parameters(pattern, upirs)
        
        return pattern
    
    def _determine_category(self, upirs: List[UPIR]) -> str:
        """Determine pattern category based on common characteristics."""
        # Check for streaming indicators
        streaming_count = 0
        batch_count = 0
        microservices_count = 0
        
        for upir in upirs:
            if upir.architecture:
                patterns = upir.architecture.patterns
                
                if "streaming" in patterns:
                    streaming_count += 1
                if "batch" in patterns:
                    batch_count += 1
                if "microservices" in patterns:
                    microservices_count += 1
                
                # Also check components
                for comp in upir.architecture.components:
                    comp_str = str(comp).lower()
                    if "stream" in comp_str or "kafka" in comp_str or "pubsub" in comp_str:
                        streaming_count += 0.5
                    if "batch" in comp_str or "spark" in comp_str:
                        batch_count += 0.5
        
        # Return most common category
        if streaming_count >= batch_count and streaming_count >= microservices_count:
            return "streaming"
        elif batch_count >= microservices_count:
            return "batch"
        else:
            return "microservices"
    
    def _extract_parameters(self, pattern: ArchitecturalPattern, upirs: List[UPIR]) -> None:
        """Extract parameters and their constraints from instances."""
        # Find all parameterized values in template
        param_values = defaultdict(list)
        
        for comp in pattern.template_components:
            if "config" in comp:
                for key, value in comp["config"].items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        param_name = value[2:-1]
                        
                        # Collect actual values from instances
                        for upir in upirs:
                            if upir.architecture:
                                for arch_comp in upir.architecture.components:
                                    if arch_comp.get("type") == comp["type"] and "config" in arch_comp:
                                        if key in arch_comp["config"]:
                                            param_values[param_name].append(arch_comp["config"][key])
        
        # Determine constraints for each parameter
        for param_name, values in param_values.items():
            if values:
                pattern.parameters[param_name] = {
                    "type": type(values[0]).__name__,
                    "default": np.median(values) if all(isinstance(v, (int, float)) for v in values) else values[0]
                }
                
                if all(isinstance(v, (int, float)) for v in values):
                    pattern.constraints[param_name] = {
                        "min": min(values),
                        "max": max(values)
                    }