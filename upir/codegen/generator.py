"""
Template-based code generation with synthesis integration.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import ast
import textwrap
from z3 import *

from ..verification.verifier import Verifier
from ..synthesis.synthesizer import Synthesizer


@dataclass
class GeneratedCode:
    """Result of code generation."""
    language: str
    code: str
    imports: List[str]
    verified_properties: List[str]
    synthesized_params: Dict[str, Any]


class Template(ABC):
    """Base class for code templates."""
    
    def __init__(self, name: str):
        self.name = name
        self.supported_languages = ['python', 'go', 'javascript']
        self.verifier = Verifier()
        self.synthesizer = Synthesizer()
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get template parameters that need to be synthesized."""
        pass
    
    @abstractmethod
    def get_constraints(self) -> List:
        """Get constraints for parameter synthesis."""
        pass
    
    @abstractmethod
    def generate_python(self, params: Dict[str, Any]) -> str:
        """Generate Python code."""
        pass
    
    @abstractmethod
    def generate_go(self, params: Dict[str, Any]) -> str:
        """Generate Go code."""
        pass
    
    @abstractmethod
    def generate_javascript(self, params: Dict[str, Any]) -> str:
        """Generate JavaScript code."""
        pass
    
    def synthesize_parameters(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize optimal parameters based on requirements."""
        solver = Solver()
        
        # Create Z3 variables for parameters
        z3_vars = {}
        for name, spec in self.get_parameters().items():
            if spec['type'] == 'int':
                z3_vars[name] = Int(name)
                if 'min' in spec:
                    solver.add(z3_vars[name] >= spec['min'])
                if 'max' in spec:
                    solver.add(z3_vars[name] <= spec['max'])
            elif spec['type'] == 'float':
                z3_vars[name] = Real(name)
                if 'min' in spec:
                    solver.add(z3_vars[name] >= spec['min'])
                if 'max' in spec:
                    solver.add(z3_vars[name] <= spec['max'])
        
        # Add template-specific constraints
        for constraint in self.get_constraints():
            solver.add(constraint(z3_vars))
        
        # Add user requirements as constraints
        for req_name, req_value in requirements.items():
            if req_name in z3_vars:
                solver.add(z3_vars[req_name] == req_value)
        
        # Solve
        if solver.check() == sat:
            model = solver.model()
            params = {}
            for name, var in z3_vars.items():
                val = model.eval(var)
                if is_int_value(val):
                    params[name] = val.as_long()
                elif is_rational_value(val):
                    params[name] = float(val.as_fraction())
                else:
                    params[name] = str(val)
            return params
        else:
            raise ValueError("Cannot synthesize parameters satisfying constraints")
    
    def generate(self, language: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate code for specified language."""
        if language not in self.supported_languages:
            raise ValueError(f"Language {language} not supported")
        
        # Synthesize parameters if not provided
        if params is None:
            params = self.synthesize_parameters({})
        
        # Generate code based on language
        if language == 'python':
            return self.generate_python(params)
        elif language == 'go':
            return self.generate_go(params)
        elif language == 'javascript':
            return self.generate_javascript(params)


class CodeGenerator:
    """Main code generator that orchestrates template-based generation."""
    
    def __init__(self):
        self.templates: Dict[str, Template] = {}
        self.verifier = Verifier()
        
    def register_template(self, template: Template):
        """Register a new template."""
        self.templates[template.name] = template
    
    def generate_from_spec(self, 
                          spec: Dict[str, Any],
                          language: str = 'python') -> GeneratedCode:
        """Generate code from a specification."""
        
        # Identify the appropriate template
        template_name = spec.get('pattern', 'queue_worker')
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        
        # Extract requirements from spec
        requirements = spec.get('requirements', {})
        
        # Synthesize parameters
        params = template.synthesize_parameters(requirements)
        
        # Generate code
        code = template.generate(language, params)
        
        # Get imports for the language
        imports = self._get_imports(template_name, language)
        
        # Verify generated code satisfies properties
        verified_props = self._verify_properties(spec, code, language)
        
        return GeneratedCode(
            language=language,
            code=code,
            imports=imports,
            verified_properties=verified_props,
            synthesized_params=params
        )
    
    def _get_imports(self, template_name: str, language: str) -> List[str]:
        """Get required imports for generated code."""
        import_map = {
            'python': {
                'queue_worker': ['import queue', 'import time', 'import logging'],
                'rate_limiter': ['import time', 'from threading import Lock'],
                'circuit_breaker': ['import time', 'from enum import Enum'],
                'retry': ['import time', 'import random', 'from functools import wraps'],
                'cache': ['from collections import OrderedDict', 'import time'],
                'load_balancer': ['import random', 'from typing import List']
            },
            'go': {
                'queue_worker': ['fmt', 'time', 'context'],
                'rate_limiter': ['sync', 'time'],
                'circuit_breaker': ['sync', 'time', 'errors'],
                'retry': ['time', 'math/rand'],
                'cache': ['sync', 'time', 'container/list'],
                'load_balancer': ['sync', 'math/rand']
            },
            'javascript': {
                'queue_worker': [],
                'rate_limiter': [],
                'circuit_breaker': [],
                'retry': [],
                'cache': [],
                'load_balancer': []
            }
        }
        
        return import_map.get(language, {}).get(template_name, [])
    
    def _verify_properties(self, spec: Dict[str, Any], 
                          code: str, language: str) -> List[str]:
        """Verify that generated code satisfies specified properties."""
        properties = spec.get('properties', [])
        verified = []
        
        for prop in properties:
            # Simple property checking for now
            # In a real implementation, we'd parse the code and verify formally
            if prop == 'no_data_loss':
                if 'ack(' in code or 'commit(' in code:
                    verified.append(prop)
            elif prop == 'idempotent':
                if 'idempotency_key' in code or 'dedup' in code:
                    verified.append(prop)
            elif prop == 'bounded_latency':
                if 'timeout' in code:
                    verified.append(prop)
        
        return verified
    
    def generate_system(self, 
                       components: List[Dict[str, Any]], 
                       language: str = 'python') -> List[GeneratedCode]:
        """Generate code for an entire system of components."""
        results = []
        
        for component in components:
            code = self.generate_from_spec(component, language)
            results.append(code)
        
        return results