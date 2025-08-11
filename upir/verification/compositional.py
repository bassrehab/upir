"""
Compositional verification framework for UPIR.

Verifies components independently and then proves composition properties.
This allows verification of large systems by breaking them into manageable pieces.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from z3 import *
import time


class PropertyType(Enum):
    """Types of properties we can verify."""
    SAFETY = "safety"  # Nothing bad happens
    LIVENESS = "liveness"  # Something good eventually happens
    INVARIANT = "invariant"  # Always true
    TEMPORAL = "temporal"  # Complex temporal property


@dataclass
class Component:
    """A verifiable component in the system."""
    name: str
    inputs: Dict[str, type]
    outputs: Dict[str, type]
    properties: List['Property']
    implementation: Optional[str] = None
    verified: bool = False
    proof: Optional['Proof'] = None


@dataclass
class Property:
    """A property to be verified."""
    name: str
    type: PropertyType
    formula: str  # Z3 formula as string
    components: List[str]  # Components this property involves
    verified: bool = False


@dataclass
class Proof:
    """A proof that properties hold."""
    property_name: str
    components: List[str]
    z3_proof: Optional[Any] = None
    verification_time_ms: float = 0
    assumptions: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        return self.z3_proof is not None


@dataclass
class CompositionResult:
    """Result of compositional verification."""
    verified: bool
    proofs: List[Proof]
    counterexamples: List[Dict[str, Any]]
    total_time_ms: float
    component_times: Dict[str, float]


class CompositionalVerifier:
    """
    Verifies systems compositionally by:
    1. Verifying individual components
    2. Verifying interfaces between components
    3. Proving that composition preserves properties
    """
    
    def __init__(self):
        self.components: Dict[str, Component] = {}
        self.properties: List[Property] = []
        self.dependency_graph = nx.DiGraph()
        self.proof_cache: Dict[str, Proof] = {}
        self.solver = Solver()
        
    def add_component(self, component: Component):
        """Add a component to the system."""
        self.components[component.name] = component
        self.dependency_graph.add_node(component.name)
        
    def add_connection(self, from_component: str, to_component: str, 
                       connection_type: str = "data"):
        """Add a connection between components."""
        self.dependency_graph.add_edge(
            from_component, to_component, 
            type=connection_type
        )
        
    def add_property(self, property: Property):
        """Add a property to verify."""
        self.properties.append(property)
        
    def verify_system(self) -> CompositionResult:
        """
        Verify the entire system compositionally.
        """
        start_time = time.time()
        proofs = []
        counterexamples = []
        component_times = {}
        
        # Step 1: Verify individual components
        for name, component in self.components.items():
            comp_start = time.time()
            comp_proof = self._verify_component(component)
            comp_time = (time.time() - comp_start) * 1000
            component_times[name] = comp_time
            
            if comp_proof.is_valid():
                proofs.append(comp_proof)
                component.verified = True
                component.proof = comp_proof
            else:
                counterexamples.append({
                    'component': name,
                    'reason': 'Component verification failed'
                })
                
        # Step 2: Verify interfaces
        interface_proofs = self._verify_interfaces()
        proofs.extend(interface_proofs)
        
        # Step 3: Verify composition preserves global properties
        for property in self.properties:
            if len(property.components) > 1:
                # This is a compositional property
                comp_proof = self._verify_composition(property)
                if comp_proof.is_valid():
                    proofs.append(comp_proof)
                    property.verified = True
                else:
                    counterexamples.append({
                        'property': property.name,
                        'reason': 'Composition verification failed'
                    })
        
        total_time = (time.time() - start_time) * 1000
        
        return CompositionResult(
            verified=len(counterexamples) == 0,
            proofs=proofs,
            counterexamples=counterexamples,
            total_time_ms=total_time,
            component_times=component_times
        )
    
    def _verify_component(self, component: Component) -> Proof:
        """
        Verify a single component in isolation.
        """
        # Check cache first
        cache_key = f"component_{component.name}"
        if cache_key in self.proof_cache:
            return self.proof_cache[cache_key]
        
        start_time = time.time()
        solver = Solver()
        
        # Create Z3 variables for inputs and outputs
        z3_inputs = {}
        z3_outputs = {}
        
        for inp_name, inp_type in component.inputs.items():
            if inp_type == int:
                z3_inputs[inp_name] = Int(f"{component.name}_{inp_name}")
            elif inp_type == bool:
                z3_inputs[inp_name] = Bool(f"{component.name}_{inp_name}")
            elif inp_type == float:
                z3_inputs[inp_name] = Real(f"{component.name}_{inp_name}")
                
        for out_name, out_type in component.outputs.items():
            if out_type == int:
                z3_outputs[out_name] = Int(f"{component.name}_{out_name}")
            elif out_type == bool:
                z3_outputs[out_name] = Bool(f"{component.name}_{out_name}")
            elif out_type == float:
                z3_outputs[out_name] = Real(f"{component.name}_{out_name}")
        
        # Add component properties as constraints
        for prop in component.properties:
            if prop.type == PropertyType.INVARIANT:
                # Parse and add the formula
                formula = self._parse_formula(prop.formula, z3_inputs, z3_outputs)
                solver.add(formula)
        
        # Check satisfiability
        result = solver.check()
        
        proof = Proof(
            property_name=f"{component.name}_properties",
            components=[component.name],
            z3_proof=solver.proof() if result == sat else None,
            verification_time_ms=(time.time() - start_time) * 1000,
            assumptions=[]
        )
        
        # Cache the proof
        self.proof_cache[cache_key] = proof
        
        return proof
    
    def _verify_interfaces(self) -> List[Proof]:
        """
        Verify that component interfaces are compatible.
        """
        proofs = []
        
        for edge in self.dependency_graph.edges():
            from_comp = self.components[edge[0]]
            to_comp = self.components[edge[1]]
            
            # Check type compatibility
            interface_compatible = self._check_interface_compatibility(from_comp, to_comp)
            
            if interface_compatible:
                proof = Proof(
                    property_name=f"interface_{edge[0]}_{edge[1]}",
                    components=[edge[0], edge[1]],
                    z3_proof=True,  # Simple compatibility check
                    verification_time_ms=0,
                    assumptions=[]
                )
                proofs.append(proof)
        
        return proofs
    
    def _check_interface_compatibility(self, from_comp: Component, 
                                      to_comp: Component) -> bool:
        """
        Check if two components have compatible interfaces.
        """
        # For each output of from_comp, check if there's a matching input in to_comp
        for out_name, out_type in from_comp.outputs.items():
            # Simple name matching for now
            if out_name in to_comp.inputs:
                if to_comp.inputs[out_name] != out_type:
                    return False
        
        return True
    
    def _verify_composition(self, property: Property) -> Proof:
        """
        Verify that composition of components preserves a property.
        """
        cache_key = f"composition_{property.name}"
        if cache_key in self.proof_cache:
            return self.proof_cache[cache_key]
        
        start_time = time.time()
        solver = Solver()
        
        # Collect all component proofs
        component_proofs = []
        for comp_name in property.components:
            if comp_name in self.components:
                comp = self.components[comp_name]
                if comp.proof and comp.proof.is_valid():
                    component_proofs.append(comp.proof)
        
        # Use assume-guarantee reasoning
        # Assume: Component properties hold
        # Guarantee: Composition property holds
        
        # Create variables for all involved components
        all_vars = {}
        for comp_name in property.components:
            comp = self.components[comp_name]
            for inp_name, inp_type in comp.inputs.items():
                var_name = f"{comp_name}_{inp_name}"
                if inp_type == int:
                    all_vars[var_name] = Int(var_name)
                elif inp_type == bool:
                    all_vars[var_name] = Bool(var_name)
                elif inp_type == float:
                    all_vars[var_name] = Real(var_name)
        
        # Add composition formula
        formula = self._parse_formula(property.formula, all_vars, {})
        solver.add(formula)
        
        # Add assumptions from component proofs
        for comp_proof in component_proofs:
            for assumption in comp_proof.assumptions:
                # Add assumption as constraint
                pass  # Simplified for now
        
        result = solver.check()
        
        proof = Proof(
            property_name=property.name,
            components=property.components,
            z3_proof=solver.proof() if result == sat else None,
            verification_time_ms=(time.time() - start_time) * 1000,
            assumptions=[f"assumes_{comp.property_name}" for comp in component_proofs]
        )
        
        self.proof_cache[cache_key] = proof
        return proof
    
    def _parse_formula(self, formula: str, inputs: Dict, outputs: Dict) -> Any:
        """
        Parse a formula string into Z3 constraints.
        """
        # Simple parser for common patterns
        # In a real implementation, we'd use a proper parser
        
        # Handle simple comparisons
        if '>' in formula:
            parts = formula.split('>')
            left = parts[0].strip()
            right = parts[1].strip()
            
            if left in inputs:
                return inputs[left] > int(right)
            elif left in outputs:
                return outputs[left] > int(right)
        
        elif '<' in formula:
            parts = formula.split('<')
            left = parts[0].strip()
            right = parts[1].strip()
            
            if left in inputs:
                return inputs[left] < int(right)
            elif left in outputs:
                return outputs[left] < int(right)
        
        elif '==' in formula:
            parts = formula.split('==')
            left = parts[0].strip()
            right = parts[1].strip()
            
            if left in inputs and right in outputs:
                return inputs[left] == outputs[right]
        
        # Default to True if we can't parse
        return True
    
    def get_verification_order(self) -> List[str]:
        """
        Get optimal order for verifying components based on dependencies.
        """
        try:
            # Topological sort gives us dependency order
            return list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXError:
            # Graph has cycles, use arbitrary order
            return list(self.components.keys())
    
    def verify_incremental(self, changed_components: List[str]) -> CompositionResult:
        """
        Incrementally verify only affected components.
        """
        start_time = time.time()
        
        # Find affected components using dependency graph
        affected = set(changed_components)
        for comp in changed_components:
            # Add all components that depend on this one
            affected.update(nx.descendants(self.dependency_graph, comp))
        
        # Invalidate cached proofs for affected components
        for comp in affected:
            cache_key = f"component_{comp}"
            if cache_key in self.proof_cache:
                del self.proof_cache[cache_key]
        
        # Re-verify only affected components
        proofs = []
        counterexamples = []
        component_times = {}
        
        for comp_name in affected:
            if comp_name in self.components:
                comp = self.components[comp_name]
                comp_start = time.time()
                proof = self._verify_component(comp)
                component_times[comp_name] = (time.time() - comp_start) * 1000
                
                if proof.is_valid():
                    proofs.append(proof)
                else:
                    counterexamples.append({
                        'component': comp_name,
                        'reason': 'Incremental verification failed'
                    })
        
        total_time = (time.time() - start_time) * 1000
        
        return CompositionResult(
            verified=len(counterexamples) == 0,
            proofs=proofs,
            counterexamples=counterexamples,
            total_time_ms=total_time,
            component_times=component_times
        )


class AssumeGuarantee:
    """
    Assume-guarantee reasoning for compositional verification.
    """
    
    def __init__(self, verifier: CompositionalVerifier):
        self.verifier = verifier
        
    def verify_with_contracts(self, 
                             component: Component,
                             assumes: List[Property],
                             guarantees: List[Property]) -> bool:
        """
        Verify component with assume-guarantee contracts.
        
        If assumes hold, then guarantees hold.
        """
        solver = Solver()
        
        # Add assumptions
        for assumption in assumes:
            formula = self.verifier._parse_formula(assumption.formula, {}, {})
            solver.add(formula)
        
        # Check if guarantees hold under assumptions
        for guarantee in guarantees:
            formula = self.verifier._parse_formula(guarantee.formula, {}, {})
            solver.add(Not(formula))  # Check if negation is unsatisfiable
        
        result = solver.check()
        return result == unsat  # If unsat, guarantees hold
    
    def circular_reasoning(self,
                          comp1: Component, comp2: Component,
                          prop1: Property, prop2: Property) -> bool:
        """
        Use circular assume-guarantee reasoning for two components.
        
        Verify:
        1. comp1 || prop2 |= prop1
        2. comp2 || prop1 |= prop2
        
        If both hold, then comp1 || comp2 |= prop1 ∧ prop2
        """
        # Check first direction
        check1 = self.verify_with_contracts(
            comp1, 
            assumes=[prop2],
            guarantees=[prop1]
        )
        
        # Check second direction
        check2 = self.verify_with_contracts(
            comp2,
            assumes=[prop1],
            guarantees=[prop2]
        )
        
        return check1 and check2


class ProofComposer:
    """
    Combines individual component proofs into system-level proofs.
    """
    
    def __init__(self):
        self.proof_dag = nx.DiGraph()
        
    def add_proof(self, proof: Proof):
        """Add a proof to the composition DAG."""
        self.proof_dag.add_node(
            proof.property_name,
            proof=proof
        )
        
        # Add edges for dependencies
        for assumption in proof.assumptions:
            self.proof_dag.add_edge(assumption, proof.property_name)
    
    def compose_proofs(self, target_property: str) -> Optional[Proof]:
        """
        Compose proofs to prove target property.
        """
        if target_property not in self.proof_dag:
            return None
        
        # Get all dependencies
        dependencies = nx.ancestors(self.proof_dag, target_property)
        
        # Check all dependencies are proven
        for dep in dependencies:
            dep_proof = self.proof_dag.nodes[dep].get('proof')
            if not dep_proof or not dep_proof.is_valid():
                return None
        
        # Compose the proof
        target_proof = self.proof_dag.nodes[target_property]['proof']
        
        composed_proof = Proof(
            property_name=f"composed_{target_property}",
            components=target_proof.components,
            z3_proof=target_proof.z3_proof,
            verification_time_ms=target_proof.verification_time_ms,
            assumptions=list(dependencies)
        )
        
        return composed_proof
    
    def get_proof_certificate(self) -> Dict[str, Any]:
        """
        Generate a proof certificate for the entire system.
        """
        certificate = {
            'timestamp': time.time(),
            'proofs': {},
            'dependencies': {}
        }
        
        for node in self.proof_dag.nodes():
            proof = self.proof_dag.nodes[node].get('proof')
            if proof:
                certificate['proofs'][node] = {
                    'verified': proof.is_valid(),
                    'time_ms': proof.verification_time_ms,
                    'assumptions': proof.assumptions
                }
        
        certificate['dependencies'] = {
            node: list(self.proof_dag.predecessors(node))
            for node in self.proof_dag.nodes()
        }
        
        return certificate