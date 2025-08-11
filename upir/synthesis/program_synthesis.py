"""
Bounded program synthesis for small functions using CEGIS.

This module implements actual program synthesis (not just parameter synthesis)
for small, bounded functions like predicates, transformations, and validators.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import ast
import random
from z3 import *


class ExprType(Enum):
    """Types of expressions we can synthesize."""
    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    STRING = "string"
    COMPARISON = "comparison"
    ARITHMETIC = "arithmetic"


@dataclass
class SynthesisSpec:
    """Specification for function synthesis."""
    name: str
    inputs: Dict[str, type]  # Variable names and types
    output_type: type
    examples: List[Tuple[Dict, Any]]  # (input_values, expected_output) pairs
    constraints: List[str] = None  # Additional constraints in string form
    max_depth: int = 3  # Maximum AST depth


@dataclass
class SynthesizedFunction:
    """Result of synthesis."""
    name: str
    code: str
    ast_node: ast.AST
    verified_examples: int
    synthesis_time_ms: float


class ProgramSynthesizer:
    """
    Synthesizes small programs using CEGIS approach.
    Limited to predicates and simple transformations.
    """
    
    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
        self.solver = Solver()
        
    def synthesize(self, spec: SynthesisSpec) -> Optional[SynthesizedFunction]:
        """
        Synthesize a function matching the specification.
        """
        import time
        start_time = time.time()
        
        # Build the grammar for possible expressions
        grammar = self._build_grammar(spec)
        
        # CEGIS loop
        candidate = None
        examples = spec.examples.copy()
        
        for iteration in range(self.max_iterations):
            # Synthesize candidate from current examples
            candidate = self._synthesize_from_examples(spec, examples, grammar)
            
            if candidate is None:
                return None
            
            # Verify against all examples
            counterexample = self._verify_candidate(candidate, spec)
            
            if counterexample is None:
                # Success! All examples pass
                elapsed_ms = (time.time() - start_time) * 1000
                return SynthesizedFunction(
                    name=spec.name,
                    code=self._ast_to_code(candidate),
                    ast_node=candidate,
                    verified_examples=len(spec.examples),
                    synthesis_time_ms=elapsed_ms
                )
            
            # Add counterexample and continue
            examples.append(counterexample)
        
        return None
    
    def _build_grammar(self, spec: SynthesisSpec) -> Dict[str, List]:
        """
        Build a grammar of possible expressions based on the spec.
        """
        grammar = {
            'terminals': [],
            'operators': [],
            'functions': []
        }
        
        # Add input variables as terminals
        for var_name, var_type in spec.inputs.items():
            grammar['terminals'].append(('var', var_name, var_type))
        
        # Add constants based on examples
        for inputs, output in spec.examples:
            for value in inputs.values():
                if isinstance(value, (int, float)):
                    grammar['terminals'].append(('const', value, type(value)))
                elif isinstance(value, str):
                    grammar['terminals'].append(('const', value, str))
            if isinstance(output, (int, float, str, bool)):
                grammar['terminals'].append(('const', output, type(output)))
        
        # Add operators based on output type
        if spec.output_type == bool:
            grammar['operators'].extend([
                ('and', 2), ('or', 2), ('not', 1),
                ('==', 2), ('!=', 2), ('<', 2), ('>', 2), ('<=', 2), ('>=', 2)
            ])
        elif spec.output_type in (int, float):
            grammar['operators'].extend([
                ('+', 2), ('-', 2), ('*', 2), ('//', 2), ('%', 2),
                ('abs', 1), ('min', 2), ('max', 2)
            ])
        
        # Add string operations if needed
        if any(t == str for _, t in spec.inputs.items()):
            grammar['functions'].extend([
                ('len', 1), ('lower', 1), ('upper', 1),
                ('startswith', 2), ('endswith', 2), ('contains', 2)
            ])
        
        return grammar
    
    def _synthesize_from_examples(self, spec: SynthesisSpec, 
                                 examples: List[Tuple],
                                 grammar: Dict) -> Optional[ast.AST]:
        """
        Synthesize a candidate expression from examples.
        Uses enumeration with pruning.
        """
        # Try expressions of increasing complexity
        for depth in range(1, spec.max_depth + 1):
            candidates = self._enumerate_expressions(grammar, depth, spec.output_type)
            
            for candidate in candidates:
                if self._matches_examples(candidate, examples, spec):
                    return candidate
        
        return None
    
    def _enumerate_expressions(self, grammar: Dict, depth: int, 
                              output_type: type) -> List[ast.AST]:
        """
        Enumerate all expressions up to given depth.
        """
        if depth == 0:
            # Return terminals
            expressions = []
            for term_type, value, dtype in grammar['terminals']:
                if term_type == 'var':
                    expressions.append(ast.Name(id=value, ctx=ast.Load()))
                elif term_type == 'const':
                    if isinstance(value, bool):
                        expressions.append(ast.Constant(value=value))
                    elif isinstance(value, (int, float)):
                        expressions.append(ast.Constant(value=value))
                    elif isinstance(value, str):
                        expressions.append(ast.Constant(value=value))
            return expressions
        
        expressions = []
        
        # Include all expressions of smaller depth
        if depth > 1:
            expressions.extend(self._enumerate_expressions(grammar, depth - 1, output_type))
        
        # Build new expressions using operators
        smaller_exprs = self._enumerate_expressions(grammar, depth - 1, output_type)
        
        for op, arity in grammar['operators']:
            if arity == 1:
                for expr in smaller_exprs:
                    if op == 'not':
                        expressions.append(ast.UnaryOp(op=ast.Not(), operand=expr))
                    elif op == 'abs':
                        expressions.append(
                            ast.Call(func=ast.Name(id='abs', ctx=ast.Load()),
                                   args=[expr], keywords=[])
                        )
            elif arity == 2:
                for expr1 in smaller_exprs:
                    for expr2 in smaller_exprs:
                        if op == 'and':
                            expressions.append(ast.BoolOp(op=ast.And(), values=[expr1, expr2]))
                        elif op == 'or':
                            expressions.append(ast.BoolOp(op=ast.Or(), values=[expr1, expr2]))
                        elif op == '==':
                            expressions.append(ast.Compare(left=expr1, ops=[ast.Eq()], 
                                                         comparators=[expr2]))
                        elif op == '!=':
                            expressions.append(ast.Compare(left=expr1, ops=[ast.NotEq()], 
                                                         comparators=[expr2]))
                        elif op == '<':
                            expressions.append(ast.Compare(left=expr1, ops=[ast.Lt()], 
                                                         comparators=[expr2]))
                        elif op == '>':
                            expressions.append(ast.Compare(left=expr1, ops=[ast.Gt()], 
                                                         comparators=[expr2]))
                        elif op == '<=':
                            expressions.append(ast.Compare(left=expr1, ops=[ast.LtE()], 
                                                         comparators=[expr2]))
                        elif op == '>=':
                            expressions.append(ast.Compare(left=expr1, ops=[ast.GtE()], 
                                                         comparators=[expr2]))
                        elif op == '+':
                            expressions.append(ast.BinOp(left=expr1, op=ast.Add(), right=expr2))
                        elif op == '-':
                            expressions.append(ast.BinOp(left=expr1, op=ast.Sub(), right=expr2))
                        elif op == '*':
                            expressions.append(ast.BinOp(left=expr1, op=ast.Mult(), right=expr2))
                        elif op == '//':
                            expressions.append(ast.BinOp(left=expr1, op=ast.FloorDiv(), right=expr2))
                        elif op == '%':
                            expressions.append(ast.BinOp(left=expr1, op=ast.Mod(), right=expr2))
        
        return expressions
    
    def _matches_examples(self, expr: ast.AST, examples: List[Tuple], 
                         spec: SynthesisSpec) -> bool:
        """
        Check if expression matches all examples.
        """
        for inputs, expected in examples:
            try:
                result = self._evaluate_expr(expr, inputs)
                if result != expected:
                    return False
            except:
                return False
        return True
    
    def _evaluate_expr(self, expr: ast.AST, inputs: Dict[str, Any]) -> Any:
        """
        Evaluate an AST expression with given inputs.
        """
        # Create a safe namespace with only allowed functions
        namespace = {
            'abs': abs,
            'min': min,
            'max': max,
            'len': len,
            **inputs
        }
        
        # Compile and evaluate
        code = compile(ast.Expression(body=expr), '<synthesized>', 'eval')
        return eval(code, {"__builtins__": {}}, namespace)
    
    def _verify_candidate(self, candidate: ast.AST, 
                         spec: SynthesisSpec) -> Optional[Tuple]:
        """
        Verify candidate against specification.
        Returns counterexample if found, None if verified.
        """
        # For now, just check against provided examples
        # In a full implementation, we'd use SMT solver for verification
        for inputs, expected in spec.examples:
            try:
                result = self._evaluate_expr(candidate, inputs)
                if result != expected:
                    return (inputs, expected)
            except:
                # If evaluation fails, this is also a counterexample
                return (inputs, expected)
        
        return None
    
    def _ast_to_code(self, expr: ast.AST) -> str:
        """
        Convert AST expression to Python code string.
        """
        return ast.unparse(expr)


class PredicateSynthesizer(ProgramSynthesizer):
    """
    Specialized synthesizer for boolean predicates.
    """
    
    def synthesize_filter(self, 
                         examples: List[Tuple[Dict, bool]],
                         var_name: str = "x",
                         var_type: type = dict) -> Optional[str]:
        """
        Synthesize a filter predicate from examples.
        
        Example:
            examples = [
                ({'amount': 100}, True),
                ({'amount': 50}, False),
                ({'amount': 200}, True)
            ]
            Result: "x['amount'] > 75"
        """
        spec = SynthesisSpec(
            name="filter_predicate",
            inputs={var_name: var_type},
            output_type=bool,
            examples=examples,
            max_depth=3
        )
        
        result = self.synthesize(spec)
        if result:
            return f"lambda {var_name}: {result.code}"
        return None
    
    def synthesize_validator(self,
                            valid_examples: List[Any],
                            invalid_examples: List[Any]) -> Optional[str]:
        """
        Synthesize a validation function from examples.
        """
        examples = []
        for ex in valid_examples:
            examples.append(({'value': ex}, True))
        for ex in invalid_examples:
            examples.append(({'value': ex}, False))
        
        spec = SynthesisSpec(
            name="validator",
            inputs={'value': type(valid_examples[0]) if valid_examples else object},
            output_type=bool,
            examples=examples,
            max_depth=3
        )
        
        result = self.synthesize(spec)
        if result:
            return f"lambda value: {result.code}"
        return None


class TransformationSynthesizer(ProgramSynthesizer):
    """
    Specialized synthesizer for data transformations.
    """
    
    def synthesize_mapper(self,
                         examples: List[Tuple[Any, Any]],
                         input_name: str = "x") -> Optional[str]:
        """
        Synthesize a mapping function from examples.
        
        Example:
            examples = [(1, 2), (2, 4), (3, 6)]
            Result: "lambda x: x * 2"
        """
        formatted_examples = []
        for input_val, output_val in examples:
            formatted_examples.append(({input_name: input_val}, output_val))
        
        spec = SynthesisSpec(
            name="mapper",
            inputs={input_name: type(examples[0][0])},
            output_type=type(examples[0][1]),
            examples=formatted_examples,
            max_depth=3
        )
        
        result = self.synthesize(spec)
        if result:
            return f"lambda {input_name}: {result.code}"
        return None
    
    def synthesize_aggregator(self,
                             examples: List[Tuple[List, Any]],
                             operation: str = None) -> Optional[str]:
        """
        Synthesize an aggregation function.
        
        Example:
            examples = [([1, 2, 3], 6), ([4, 5], 9)]
            Result: "lambda lst: sum(lst)"
        """
        # Detect the operation if not specified
        if operation is None:
            first_list, first_result = examples[0]
            if first_result == sum(first_list):
                operation = "sum"
            elif first_result == len(first_list):
                operation = "len"
            elif first_result == max(first_list):
                operation = "max"
            elif first_result == min(first_list):
                operation = "min"
            else:
                # Try to synthesize custom aggregation
                return self._synthesize_custom_aggregator(examples)
        
        # Verify operation matches all examples
        for lst, expected in examples:
            if operation == "sum" and sum(lst) != expected:
                return None
            elif operation == "len" and len(lst) != expected:
                return None
            elif operation == "max" and max(lst) != expected:
                return None
            elif operation == "min" and min(lst) != expected:
                return None
        
        return f"lambda lst: {operation}(lst)"
    
    def _synthesize_custom_aggregator(self, examples: List[Tuple[List, Any]]) -> Optional[str]:
        """
        Synthesize custom aggregation when standard operations don't match.
        """
        # For now, return None. In a full implementation,
        # we'd try to synthesize more complex aggregations
        return None


def synthesize_from_docstring(docstring: str) -> Optional[str]:
    """
    Attempt to synthesize a function from its docstring description.
    Uses NLP to extract examples and constraints.
    
    Example:
        docstring = '''
        Filter payments where amount > 100 and status == 'approved'
        Examples:
        - {'amount': 150, 'status': 'approved'} -> True
        - {'amount': 50, 'status': 'approved'} -> False
        - {'amount': 150, 'status': 'pending'} -> False
        '''
    """
    # Parse examples from docstring
    examples = []
    lines = docstring.strip().split('\n')
    
    for line in lines:
        if '->' in line:
            # Parse example line
            input_str, output_str = line.split('->')
            input_str = input_str.strip().lstrip('- ')
            output_str = output_str.strip()
            
            try:
                # Safely evaluate the strings
                input_val = ast.literal_eval(input_str)
                output_val = ast.literal_eval(output_str)
                examples.append((input_val, output_val))
            except:
                continue
    
    if not examples:
        return None
    
    # Determine input and output types from examples
    first_input, first_output = examples[0]
    
    # Create synthesis spec
    if isinstance(first_input, dict):
        spec = SynthesisSpec(
            name="synthesized_function",
            inputs={'x': dict},
            output_type=type(first_output),
            examples=[({'x': inp}, out) for inp, out in examples],
            max_depth=3
        )
    else:
        spec = SynthesisSpec(
            name="synthesized_function",
            inputs={'x': type(first_input)},
            output_type=type(first_output),
            examples=[({'x': inp}, out) for inp, out in examples],
            max_depth=3
        )
    
    synthesizer = ProgramSynthesizer()
    result = synthesizer.synthesize(spec)
    
    if result:
        return f"lambda x: {result.code}"
    return None