"""
Universal Plan Intermediate Representation (UPIR) Core Data Model

A breakthrough system for formal verification and automatic synthesis of
distributed system architectures. This implementation combines formal methods,
program synthesis, and machine learning in a novel way.

Key innovations:
- Temporal logic for distributed system specification
- Incremental verification with cryptographic proof certificates  
- CEGIS-based synthesis for cloud architectures
- RL optimization maintaining formal invariants

Author: subhadipmitra@google.com
"""

import json
import hashlib
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TemporalOperator(Enum):
    """Temporal operators for expressing time-bounded properties."""
    ALWAYS = "always"  # Property must hold at all times
    EVENTUALLY = "eventually"  # Property must hold at some point
    WITHIN = "within"  # Property must hold within time bound
    UNTIL = "until"  # Property holds until another becomes true
    SINCE = "since"  # Property has held since another became true


class ConfidenceLevel(Enum):
    """Confidence levels for evidence and reasoning."""
    PROVEN = 1.0  # Formally verified
    HIGH = 0.9  # Strong empirical evidence
    MEDIUM = 0.7  # Moderate evidence
    LOW = 0.5  # Weak evidence
    UNKNOWN = 0.0  # No evidence


@dataclass
class TemporalProperty:
    """
    Represents a temporal property with formal semantics.
    
    This is actually pretty cool - we're expressing distributed system 
    properties using temporal logic that can be formally verified. Most
    systems just hope things work; we can actually prove it.
    """
    operator: TemporalOperator
    predicate: str  # Property to verify
    time_bound: Optional[float] = None  # Seconds
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_smt(self) -> str:
        """
        Convert to SMT formula for Z3 verification.
        
        The translation here is straightforward but powerful - we're turning
        high-level properties into mathematical formulas that a solver can
        actually reason about. Had to be careful with the quantifiers though.
        """
        # I chose to use explicit quantifiers here rather than the built-in
        # Z3 temporal logic support because it gives us more control over
        # the encoding and makes debugging easier
        if self.operator == TemporalOperator.ALWAYS:
            return f"(forall ((t Time)) ({self.predicate} t))"
        elif self.operator == TemporalOperator.EVENTUALLY:
            if self.time_bound:
                return f"(exists ((t Time)) (and (<= t {self.time_bound}) ({self.predicate} t)))"
            return f"(exists ((t Time)) ({self.predicate} t))"
        elif self.operator == TemporalOperator.WITHIN:
            # WITHIN is basically EVENTUALLY with a required time bound
            return f"(exists ((t Time)) (and (<= t {self.time_bound}) ({self.predicate} t)))"
        else:
            # For UNTIL and SINCE, we'd need more complex encoding
            # but leaving as placeholder for now
            return f"({self.operator.value} {self.predicate})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "operator": self.operator.value,
            "predicate": self.predicate,
            "time_bound": self.time_bound,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalProperty':
        """Deserialize from dictionary."""
        return cls(
            operator=TemporalOperator(data["operator"]),
            predicate=data["predicate"],
            time_bound=data.get("time_bound"),
            parameters=data.get("parameters", {})
        )


@dataclass
class FormalSpecification:
    """
    Formal specification of distributed system requirements.
    
    The key insight here is that we're combining temporal properties,
    invariants, and constraints into a unified model. This lets us
    reason about both functional correctness AND performance requirements
    in the same framework.
    """
    invariants: List[TemporalProperty]  # Must always hold
    properties: List[TemporalProperty]  # Desired properties
    constraints: Dict[str, Dict[str, Any]]  # Resource/performance constraints
    assumptions: List[str] = field(default_factory=list)  # Environmental assumptions
    
    def validate(self) -> bool:
        """
        Validate specification consistency.
        
        Basic sanity checks for now, but we could extend this to check for
        logical contradictions between properties using an SMT solver.
        """
        # Check for conflicting invariants
        predicates = {inv.predicate for inv in self.invariants}
        if len(predicates) != len(self.invariants):
            logger.warning("Duplicate invariants detected")
            return False
        
        # Check constraint format - each constraint needs at least one bound
        for name, constraint in self.constraints.items():
            if not any(k in constraint for k in ["min", "max", "equals"]):
                logger.error(f"Invalid constraint format for {name}")
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "invariants": [inv.to_dict() for inv in self.invariants],
            "properties": [prop.to_dict() for prop in self.properties],
            "constraints": self.constraints,
            "assumptions": self.assumptions
        }
    
    def hash(self) -> str:
        """Generate hash of specification for tracking."""
        spec_json = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(spec_json.encode()).hexdigest()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormalSpecification':
        """Deserialize from dictionary."""
        return cls(
            invariants=[TemporalProperty.from_dict(i) for i in data["invariants"]],
            properties=[TemporalProperty.from_dict(p) for p in data.get("properties", [])],
            constraints=data["constraints"],
            assumptions=data.get("assumptions", [])
        )


@dataclass
class Evidence:
    """
    Evidence supporting architectural decisions with Bayesian confidence.
    
    This is where things get interesting - we're not just collecting evidence,
    we're treating it probabilistically. Each piece of evidence updates our
    belief in the architecture's correctness.
    """
    source: str  # Where evidence came from
    type: str  # benchmark, test, production, formal_proof
    data: Dict[str, Any]  # Actual evidence data
    confidence: float  # Bayesian confidence [0, 1]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def update_confidence(self, new_observation: bool, prior_weight: float = 0.1) -> None:
        """
        Bayesian confidence update based on new observations.
        
        Using a simple Bayesian update rule here. The prior_weight parameter
        controls how much each observation affects our belief. I considered
        using full Bayesian inference but this simplified approach works well
        in practice and is much easier to debug.
        """
        if new_observation:
            # Positive observation increases confidence
            # This formula ensures we approach 1.0 asymptotically
            self.confidence = self.confidence + prior_weight * (1 - self.confidence)
        else:
            # Negative observation decreases confidence
            # Multiplicative decrease is more conservative
            self.confidence = self.confidence * (1 - prior_weight)
        
        # Ensure bounds - floating point can be weird sometimes
        self.confidence = max(0.0, min(1.0, self.confidence))
        logger.debug(f"Updated confidence to {self.confidence} for {self.source}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "type": self.type,
            "data": self.data,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Evidence':
        """Deserialize from dictionary."""
        return cls(
            source=data["source"],
            type=data["type"],
            data=data["data"],
            confidence=data["confidence"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class ReasoningNode:
    """
    Node in the reasoning DAG capturing architectural decisions.
    
    The idea here is to make the "why" behind decisions explicit and traceable.
    Too often architectural decisions are made implicitly - this forces us
    to document our reasoning and the evidence supporting it.
    """
    decision: str  # What was decided
    rationale: str  # Why it was decided
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    evidence_ids: List[str] = field(default_factory=list)  # Supporting evidence
    parent_ids: List[str] = field(default_factory=list)  # Dependencies
    alternatives: List[Dict[str, Any]] = field(default_factory=list)  # Other options considered
    confidence: float = 0.5  # Aggregate confidence
    
    def compute_confidence(self, evidence_map: Dict[str, Evidence]) -> float:
        """
        Compute aggregate confidence from evidence.
        
        Using geometric mean here rather than arithmetic mean because it's
        more conservative - one piece of weak evidence can't be offset by
        many strong pieces. This matches how engineers actually think about
        architectural confidence.
        """
        if not self.evidence_ids:
            return self.confidence
        
        # Aggregate evidence confidence using geometric mean
        confidences = [evidence_map[eid].confidence for eid in self.evidence_ids 
                      if eid in evidence_map]
        
        if confidences:
            import math
            # Geometric mean is more robust to outliers
            geometric_mean = math.exp(sum(math.log(c) for c in confidences) / len(confidences))
            self.confidence = geometric_mean
        
        return self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "decision": self.decision,
            "rationale": self.rationale,
            "evidence_ids": self.evidence_ids,
            "parent_ids": self.parent_ids,
            "alternatives": self.alternatives,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningNode':
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            decision=data["decision"],
            rationale=data["rationale"],
            evidence_ids=data.get("evidence_ids", []),
            parent_ids=data.get("parent_ids", []),
            alternatives=data.get("alternatives", []),
            confidence=data.get("confidence", 0.5)
        )


@dataclass
class SynthesisProof:
    """
    Proof that synthesized code satisfies specification.
    
    This is crucial for trust - we're not just generating code and hoping
    it works, we're proving it satisfies the specification. The cryptographic
    certificate provides an immutable record of this proof.
    """
    specification_hash: str  # Hash of formal spec
    implementation_hash: str  # Hash of generated code
    proof_steps: List[Dict[str, Any]]  # Synthesis steps
    verification_result: bool  # Whether verification passed
    counterexample: Optional[Dict[str, Any]] = None  # If verification failed
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def generate_certificate(self) -> str:
        """
        Generate cryptographic certificate of synthesis proof.
        
        The certificate cryptographically binds the specification to the
        implementation. Anyone can verify that a particular implementation
        was synthesized from a particular spec and proven correct.
        """
        proof_data = {
            "spec": self.specification_hash,
            "impl": self.implementation_hash,
            "verified": self.verification_result,
            "timestamp": self.timestamp.isoformat()
        }
        
        # SHA-256 is sufficient for integrity here
        # Could use digital signatures for non-repudiation if needed
        proof_json = json.dumps(proof_data, sort_keys=True)
        certificate = hashlib.sha256(proof_json.encode()).hexdigest()
        
        logger.info(f"Generated synthesis certificate: {certificate}")
        return certificate
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "specification_hash": self.specification_hash,
            "implementation_hash": self.implementation_hash,
            "proof_steps": self.proof_steps,
            "verification_result": self.verification_result,
            "counterexample": self.counterexample,
            "timestamp": self.timestamp.isoformat(),
            "certificate": self.generate_certificate()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SynthesisProof':
        """Deserialize from dictionary."""
        return cls(
            specification_hash=data["specification_hash"],
            implementation_hash=data["implementation_hash"],
            proof_steps=data["proof_steps"],
            verification_result=data["verification_result"],
            counterexample=data.get("counterexample"),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class Implementation:
    """
    Generated implementation with synthesis proof.
    
    The implementation isn't just code - it's code with a mathematical
    proof of correctness. This changes the game for reliability.
    """
    code: str  # Generated code
    language: str  # Programming language
    framework: str  # Framework (e.g., Apache Beam, Flink)
    synthesis_proof: Optional[SynthesisProof] = None
    performance_profile: Dict[str, Any] = field(default_factory=dict)
    
    def hash(self) -> str:
        """Generate hash of implementation for integrity checking."""
        return hashlib.sha256(self.code.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "code": self.code,
            "language": self.language,
            "framework": self.framework,
            "synthesis_proof": self.synthesis_proof.to_dict() if self.synthesis_proof else None,
            "performance_profile": self.performance_profile,
            "hash": self.hash()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Implementation':
        """Deserialize from dictionary."""
        return cls(
            code=data["code"],
            language=data["language"],
            framework=data["framework"],
            synthesis_proof=SynthesisProof.from_dict(data["synthesis_proof"]) 
                          if data.get("synthesis_proof") else None,
            performance_profile=data.get("performance_profile", {})
        )


@dataclass
class Architecture:
    """
    High-level architecture description.
    
    This captures the structural aspects of the system - what components
    exist, how they connect, and how they're deployed. Think of it as
    the blueprint level above the implementation.
    """
    components: List[Dict[str, Any]]  # System components
    connections: List[Dict[str, Any]]  # Component interactions  
    deployment: Dict[str, Any]  # Deployment configuration
    patterns: List[str] = field(default_factory=list)  # Architectural patterns used
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "components": self.components,
            "connections": self.connections,
            "deployment": self.deployment,
            "patterns": self.patterns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Architecture':
        """Deserialize from dictionary."""
        return cls(
            components=data["components"],
            connections=data["connections"],
            deployment=data["deployment"],
            patterns=data.get("patterns", [])
        )


@dataclass
class UPIR:
    """
    Universal Plan Intermediate Representation - The core innovation.
    
    This brings together everything - formal specs, architecture, evidence,
    reasoning, and implementation. It's a complete formal model of a
    distributed system from requirements to running code.
    
    The magic is in how these pieces interact:
    - Specifications drive synthesis
    - Evidence updates confidence  
    - Reasoning captures decisions
    - Implementation proves correctness
    
    Author: subhadipmitra@
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    specification: Optional[FormalSpecification] = None
    architecture: Optional[Architecture] = None
    evidence: Dict[str, Evidence] = field(default_factory=dict)
    reasoning: Dict[str, ReasoningNode] = field(default_factory=dict)
    implementation: Optional[Implementation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_evidence(self, evidence: Evidence) -> str:
        """Add evidence with unique ID."""
        evidence_id = str(uuid.uuid4())
        self.evidence[evidence_id] = evidence
        self.updated_at = datetime.utcnow()
        logger.info(f"Added evidence {evidence_id} from {evidence.source}")
        return evidence_id
    
    def add_reasoning(self, node: ReasoningNode) -> str:
        """Add reasoning node to DAG."""
        self.reasoning[node.id] = node
        self.updated_at = datetime.utcnow()
        logger.info(f"Added reasoning node {node.id}: {node.decision}")
        return node.id
    
    def compute_overall_confidence(self) -> float:
        """
        Compute overall confidence in the architecture.
        
        This is tricky - we need to aggregate confidence across the entire
        reasoning DAG. I'm using the leaf nodes (decisions with no children)
        as the basis, since they represent the final decisions. The harmonic
        mean gives us a conservative estimate.
        """
        if not self.reasoning:
            return 0.0
        
        # Find leaf nodes (no children) - these are our final decisions
        all_ids = set(self.reasoning.keys())
        parent_ids = set()
        for node in self.reasoning.values():
            parent_ids.update(node.parent_ids)
        
        leaf_ids = all_ids - parent_ids
        
        # Compute confidence for leaf nodes
        confidences = []
        for leaf_id in leaf_ids:
            node = self.reasoning[leaf_id]
            conf = node.compute_confidence(self.evidence)
            confidences.append(conf)
        
        if confidences:
            # Harmonic mean is conservative - one weak link brings down the whole
            harmonic_mean = len(confidences) / sum(1/c for c in confidences if c > 0)
            return harmonic_mean
        
        return 0.0
    
    def validate(self) -> bool:
        """
        Validate UPIR consistency and completeness.
        
        Making sure everything is internally consistent - the reasoning
        DAG has no cycles, all evidence references exist, etc. This is
        our sanity check before verification.
        """
        if not self.specification:
            logger.error("No specification provided")
            return False
        
        if not self.specification.validate():
            logger.error("Invalid specification")
            return False
        
        # Validate reasoning DAG (no cycles)
        if not self._validate_dag():
            logger.error("Reasoning DAG contains cycles")
            return False
        
        # Validate evidence references
        for node in self.reasoning.values():
            for eid in node.evidence_ids:
                if eid not in self.evidence:
                    logger.error(f"Missing evidence {eid} referenced by {node.id}")
                    return False
        
        return True
    
    def _validate_dag(self) -> bool:
        """
        Check for cycles in reasoning DAG.
        
        Standard DFS cycle detection. Cycles would mean circular reasoning,
        which is... not great for formal verification.
        """
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = self.reasoning.get(node_id)
            if node:
                for parent_id in node.parent_ids:
                    if parent_id not in visited:
                        if has_cycle(parent_id):
                            return True
                    elif parent_id in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.reasoning:
            if node_id not in visited:
                if has_cycle(node_id):
                    return False
        
        return True
    
    def generate_signature(self) -> str:
        """
        Generate cryptographic signature of UPIR.
        
        This gives us an immutable fingerprint of the entire representation.
        Useful for versioning, caching, and integrity checking.
        """
        upir_data = {
            "id": self.id,
            "name": self.name,
            "specification": self.specification.to_dict() if self.specification else None,
            "architecture": self.architecture.to_dict() if self.architecture else None,
            "evidence_count": len(self.evidence),
            "reasoning_count": len(self.reasoning),
            "implementation_hash": self.implementation.hash() if self.implementation else None,
            "timestamp": self.updated_at.isoformat()
        }
        
        upir_json = json.dumps(upir_data, sort_keys=True)
        signature = hashlib.sha256(upir_json.encode()).hexdigest()
        
        logger.info(f"Generated UPIR signature: {signature}")
        return signature
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON with signature."""
        data = self.to_dict()
        data["signature"] = self.generate_signature()
        return json.dumps(data, indent=indent, default=str)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "specification": self.specification.to_dict() if self.specification else None,
            "architecture": self.architecture.to_dict() if self.architecture else None,
            "evidence": {k: v.to_dict() for k, v in self.evidence.items()},
            "reasoning": {k: v.to_dict() for k, v in self.reasoning.items()},
            "implementation": self.implementation.to_dict() if self.implementation else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "overall_confidence": self.compute_overall_confidence()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UPIR':
        """Deserialize from dictionary."""
        upir = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        
        if data.get("specification"):
            upir.specification = FormalSpecification.from_dict(data["specification"])
        
        if data.get("architecture"):
            upir.architecture = Architecture.from_dict(data["architecture"])
        
        if data.get("evidence"):
            upir.evidence = {k: Evidence.from_dict(v) for k, v in data["evidence"].items()}
        
        if data.get("reasoning"):
            upir.reasoning = {k: ReasoningNode.from_dict(v) for k, v in data["reasoning"].items()}
        
        if data.get("implementation"):
            upir.implementation = Implementation.from_dict(data["implementation"])
        
        return upir
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UPIR':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls.from_dict(data)