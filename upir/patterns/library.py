"""
Pattern Library and Recommendation System

This module manages a library of discovered architectural patterns,
provides search and recommendation capabilities, and tracks pattern
success rates over time.

The pattern library enables:
1. Storage and retrieval of patterns
2. Similarity-based search
3. Recommendation based on requirements
4. Pattern evolution and improvement

Author: subhadipmitra@google.com
"""

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import logging

from .extractor import (
    ArchitecturalPattern, PatternFeatures, FeatureExtractor,
    PatternClusterer, PatternAbstractor
)
from ..core.models import UPIR, Architecture, FormalSpecification

logger = logging.getLogger(__name__)


@dataclass
class PatternUsage:
    """Tracks usage of a pattern."""
    pattern_id: str
    upir_id: str
    timestamp: datetime
    success: bool
    performance_metrics: Dict[str, float]
    feedback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "upir_id": self.upir_id,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "performance_metrics": self.performance_metrics,
            "feedback": self.feedback
        }


@dataclass
class SearchQuery:
    """Query for pattern search."""
    requirements: Optional[FormalSpecification] = None
    component_types: Optional[List[str]] = None
    patterns: Optional[List[str]] = None  # Pattern names to include
    min_success_rate: float = 0.7
    max_results: int = 10
    category: Optional[str] = None


class PatternLibrary:
    """
    Central library for architectural patterns.
    
    This is like a cookbook for distributed systems - proven recipes
    that can be adapted for specific needs.
    """
    
    def __init__(self, storage_path: str = None):
        """
        Initialize pattern library.
        
        Args:
            storage_path: Path to persist patterns (optional)
        """
        self.patterns: Dict[str, ArchitecturalPattern] = {}
        self.usage_history: List[PatternUsage] = []
        self.feature_extractor = FeatureExtractor()
        self.pattern_features: Dict[str, PatternFeatures] = {}
        self.storage_path = storage_path
        
        # Load existing patterns if storage path provided
        if storage_path and os.path.exists(storage_path):
            self.load()
    
    def add_pattern(self, pattern: ArchitecturalPattern) -> str:
        """
        Add a pattern to the library.
        
        Returns pattern ID.
        """
        self.patterns[pattern.id] = pattern
        
        # Extract features for similarity search
        # Create a dummy UPIR from pattern for feature extraction
        dummy_arch = pattern.instantiate({})
        dummy_upir = UPIR(name=pattern.name)
        dummy_upir.architecture = dummy_arch
        
        try:
            features = self.feature_extractor.extract(dummy_upir)
            self.pattern_features[pattern.id] = features
        except Exception as e:
            logger.warning(f"Could not extract features for pattern {pattern.id}: {e}")
        
        logger.info(f"Added pattern {pattern.name} to library")
        
        # Persist if storage path set
        if self.storage_path:
            self.save()
        
        return pattern.id
    
    def get_pattern(self, pattern_id: str) -> Optional[ArchitecturalPattern]:
        """Get a pattern by ID."""
        return self.patterns.get(pattern_id)
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove a pattern from the library."""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
            if pattern_id in self.pattern_features:
                del self.pattern_features[pattern_id]
            
            logger.info(f"Removed pattern {pattern_id} from library")
            
            if self.storage_path:
                self.save()
            
            return True
        return False
    
    def update_pattern(self, pattern_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update pattern properties.
        
        This allows patterns to evolve based on usage.
        """
        if pattern_id not in self.patterns:
            return False
        
        pattern = self.patterns[pattern_id]
        
        # Update allowed fields
        if "name" in updates:
            pattern.name = updates["name"]
        if "description" in updates:
            pattern.description = updates["description"]
        if "parameters" in updates:
            pattern.parameters.update(updates["parameters"])
        if "constraints" in updates:
            pattern.constraints.update(updates["constraints"])
        
        logger.info(f"Updated pattern {pattern_id}")
        
        if self.storage_path:
            self.save()
        
        return True
    
    def search(self, query: SearchQuery) -> List[Tuple[ArchitecturalPattern, float]]:
        """
        Search for patterns matching query.
        
        Returns list of (pattern, relevance_score) tuples.
        """
        results = []
        
        for pattern_id, pattern in self.patterns.items():
            # Check success rate threshold
            if pattern.success_rate < query.min_success_rate:
                continue
            
            # Check category if specified
            if query.category and pattern.category != query.category:
                continue
            
            # Calculate relevance score
            score = self._calculate_relevance(pattern, query)
            
            if score > 0:
                results.append((pattern, score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Limit results
        return results[:query.max_results]
    
    def recommend(self, upir: UPIR, top_k: int = 5) -> List[Tuple[ArchitecturalPattern, float]]:
        """
        Recommend patterns for a UPIR based on its requirements.
        
        This is where the magic happens - we match requirements to
        proven patterns that can satisfy them.
        """
        if not upir.specification:
            return []
        
        # Extract features from UPIR if it has an architecture
        upir_features = None
        if upir.architecture:
            try:
                upir_features = self.feature_extractor.extract(upir)
            except:
                pass
        
        recommendations = []
        
        for pattern_id, pattern in self.patterns.items():
            # Calculate compatibility score
            score = 0.0
            
            # Check if pattern satisfies required properties
            if upir.specification.invariants:
                required_props = {p.predicate for p in upir.specification.invariants}
                pattern_props = set(pattern.required_properties)
                
                # Proportion of requirements satisfied
                if required_props:
                    prop_score = len(required_props & pattern_props) / len(required_props)
                    score += prop_score * 0.4
            
            # Feature similarity if both have features
            if upir_features and pattern_id in self.pattern_features:
                pattern_features = self.pattern_features[pattern_id]
                similarity = self._calculate_feature_similarity(upir_features, pattern_features)
                score += similarity * 0.3
            
            # Success rate contributes to score
            score += pattern.success_rate * 0.2
            
            # Recency bonus (prefer recently used patterns)
            if pattern.last_used:
                days_ago = (datetime.utcnow() - pattern.last_used).days
                recency_score = max(0, 1 - days_ago / 30)  # Decay over 30 days
                score += recency_score * 0.1
            
            if score > 0:
                recommendations.append((pattern, score))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:top_k]
    
    def record_usage(self, pattern_id: str, upir_id: str, 
                    success: bool, performance: Dict[str, float],
                    feedback: str = None) -> None:
        """
        Record pattern usage for tracking and learning.
        
        This helps patterns improve over time based on real usage.
        """
        usage = PatternUsage(
            pattern_id=pattern_id,
            upir_id=upir_id,
            timestamp=datetime.utcnow(),
            success=success,
            performance_metrics=performance,
            feedback=feedback
        )
        
        self.usage_history.append(usage)
        
        # Update pattern statistics
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.update_statistics(success, performance)
            
            # Add to instances if successful
            if success and upir_id not in pattern.instances:
                pattern.instances.append(upir_id)
        
        logger.info(f"Recorded usage of pattern {pattern_id}: success={success}")
        
        if self.storage_path:
            self.save()
    
    def discover_patterns(self, upirs: List[UPIR], 
                         min_cluster_size: int = 3) -> List[ArchitecturalPattern]:
        """
        Discover new patterns from a collection of UPIRs.
        
        This is how the library grows - by learning from successful
        architectures in production.
        """
        if len(upirs) < min_cluster_size:
            logger.warning(f"Need at least {min_cluster_size} UPIRs to discover patterns")
            return []
        
        discovered = []
        
        # Extract features from all UPIRs
        features_list = []
        valid_upirs = []
        
        for upir in upirs:
            if upir.architecture:
                try:
                    features = self.feature_extractor.extract(upir)
                    features_list.append(features)
                    valid_upirs.append(upir)
                except Exception as e:
                    logger.warning(f"Could not extract features from {upir.id}: {e}")
        
        if len(valid_upirs) < min_cluster_size:
            return []
        
        # Cluster architectures
        clusterer = PatternClusterer(method="dbscan")
        clusters = clusterer.cluster(features_list, min_cluster_size)
        
        logger.info(f"Found {len(clusters)} clusters from {len(valid_upirs)} UPIRs")
        
        # Abstract each cluster into a pattern
        abstractor = PatternAbstractor()
        
        for cluster_id, indices in clusters.items():
            cluster_upirs = [valid_upirs[i] for i in indices]
            
            try:
                # Create pattern from cluster
                pattern = abstractor.abstract(
                    cluster_upirs,
                    name=f"discovered_pattern_{cluster_id}"
                )
                
                # Add to library
                self.add_pattern(pattern)
                discovered.append(pattern)
                
                logger.info(f"Discovered pattern {pattern.name} from {len(cluster_upirs)} instances")
                
            except Exception as e:
                logger.error(f"Failed to abstract cluster {cluster_id}: {e}")
        
        return discovered
    
    def evolve_patterns(self, min_usage: int = 10) -> List[str]:
        """
        Evolve patterns based on usage history.
        
        Patterns that consistently fail are deprecated, while
        successful patterns are refined based on what works.
        """
        evolved = []
        
        # Analyze usage by pattern
        pattern_stats = defaultdict(lambda: {"success": 0, "total": 0, "metrics": []})
        
        for usage in self.usage_history:
            stats = pattern_stats[usage.pattern_id]
            stats["total"] += 1
            if usage.success:
                stats["success"] += 1
            stats["metrics"].append(usage.performance_metrics)
        
        for pattern_id, stats in pattern_stats.items():
            if stats["total"] < min_usage:
                continue  # Not enough data
            
            pattern = self.patterns.get(pattern_id)
            if not pattern:
                continue
            
            success_rate = stats["success"] / stats["total"]
            
            # Deprecate patterns with low success rate
            if success_rate < 0.3:
                pattern.description = f"[DEPRECATED] {pattern.description}"
                logger.warning(f"Deprecated pattern {pattern.name} due to low success rate")
                evolved.append(pattern_id)
            
            # Refine successful patterns
            elif success_rate > 0.8:
                # Update constraints based on successful usage
                if stats["metrics"]:
                    # Analyze what parameter values lead to success
                    # This is simplified - real implementation would be more sophisticated
                    avg_metrics = {}
                    for metrics in stats["metrics"]:
                        for key, value in metrics.items():
                            if key not in avg_metrics:
                                avg_metrics[key] = []
                            avg_metrics[key].append(value)
                    
                    # Update pattern's average performance
                    for key, values in avg_metrics.items():
                        pattern.avg_performance[key] = np.mean(values)
                
                logger.info(f"Refined pattern {pattern.name} based on {stats['total']} uses")
                evolved.append(pattern_id)
        
        if self.storage_path:
            self.save()
        
        return evolved
    
    def _calculate_relevance(self, pattern: ArchitecturalPattern, 
                           query: SearchQuery) -> float:
        """Calculate relevance score for a pattern given a query."""
        score = 0.0
        
        # Check component types if specified
        if query.component_types:
            pattern_types = {c["type"] for c in pattern.template_components if "type" in c}
            query_types = set(query.component_types)
            
            if pattern_types & query_types:
                overlap = len(pattern_types & query_types) / len(query_types)
                score += overlap * 0.4
        
        # Check pattern names if specified
        if query.patterns:
            if pattern.name in query.patterns or pattern.category in query.patterns:
                score += 0.3
        
        # Check requirements satisfaction
        if query.requirements:
            required_props = {p.predicate for p in query.requirements.invariants}
            if required_props:
                satisfied = len(required_props & set(pattern.required_properties))
                score += (satisfied / len(required_props)) * 0.3
        
        # Success rate bonus
        score += pattern.success_rate * 0.2
        
        return score
    
    def _calculate_feature_similarity(self, features1: PatternFeatures,
                                    features2: PatternFeatures) -> float:
        """Calculate similarity between two feature sets."""
        # Convert to vectors
        v1 = features1.to_vector()
        v2 = features2.to_vector()
        
        # Cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 * norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Normalize to [0, 1]
        return (similarity + 1) / 2
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        stats = {
            "total_patterns": len(self.patterns),
            "total_usage": len(self.usage_history),
            "categories": defaultdict(int),
            "avg_success_rate": 0.0,
            "most_used": None,
            "most_successful": None
        }
        
        # Category distribution
        for pattern in self.patterns.values():
            stats["categories"][pattern.category] += 1
        
        # Average success rate
        if self.patterns:
            success_rates = [p.success_rate for p in self.patterns.values()]
            stats["avg_success_rate"] = np.mean(success_rates)
        
        # Most used pattern
        usage_counts = defaultdict(int)
        for usage in self.usage_history:
            usage_counts[usage.pattern_id] += 1
        
        if usage_counts:
            most_used_id = max(usage_counts, key=usage_counts.get)
            if most_used_id in self.patterns:
                stats["most_used"] = self.patterns[most_used_id].name
        
        # Most successful pattern
        if self.patterns:
            most_successful = max(self.patterns.values(), key=lambda p: p.success_rate)
            stats["most_successful"] = most_successful.name
        
        return stats
    
    def save(self) -> None:
        """Save library to storage."""
        if not self.storage_path:
            return
        
        data = {
            "patterns": {pid: p.to_dict() for pid, p in self.patterns.items()},
            "usage_history": [u.to_dict() for u in self.usage_history[-1000:]],  # Keep last 1000
            "pattern_features": {}  # Features are recomputed on load
        }
        
        # Save as JSON
        json_path = os.path.join(self.storage_path, "pattern_library.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved pattern library to {json_path}")
    
    def load(self) -> None:
        """Load library from storage."""
        if not self.storage_path:
            return
        
        json_path = os.path.join(self.storage_path, "pattern_library.json")
        
        if not os.path.exists(json_path):
            logger.warning(f"No pattern library found at {json_path}")
            return
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Restore patterns
        self.patterns = {}
        for pid, pdata in data.get("patterns", {}).items():
            # Reconstruct pattern
            pattern = ArchitecturalPattern(
                id=pdata["id"],
                name=pdata["name"],
                description=pdata["description"],
                category=pdata["category"],
                template_components=pdata["template_components"],
                template_connections=pdata["template_connections"],
                required_properties=pdata["required_properties"],
                instances=pdata.get("instances", []),
                success_rate=pdata.get("success_rate", 0.5),
                avg_performance=pdata.get("avg_performance", {}),
                parameters=pdata.get("parameters", {}),
                constraints=pdata.get("constraints", {})
            )
            
            if "discovered_at" in pdata:
                pattern.discovered_at = datetime.fromisoformat(pdata["discovered_at"])
            if pdata.get("last_used"):
                pattern.last_used = datetime.fromisoformat(pdata["last_used"])
            
            self.patterns[pid] = pattern
        
        # Restore usage history
        self.usage_history = []
        for udata in data.get("usage_history", []):
            usage = PatternUsage(
                pattern_id=udata["pattern_id"],
                upir_id=udata["upir_id"],
                timestamp=datetime.fromisoformat(udata["timestamp"]),
                success=udata["success"],
                performance_metrics=udata["performance_metrics"],
                feedback=udata.get("feedback")
            )
            self.usage_history.append(usage)
        
        # Recompute features
        for pattern in self.patterns.values():
            try:
                dummy_arch = pattern.instantiate({})
                dummy_upir = UPIR(name=pattern.name)
                dummy_upir.architecture = dummy_arch
                features = self.feature_extractor.extract(dummy_upir)
                self.pattern_features[pattern.id] = features
            except:
                pass
        
        logger.info(f"Loaded {len(self.patterns)} patterns from {json_path}")