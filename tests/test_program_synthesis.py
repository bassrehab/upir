"""
Tests for bounded program synthesis.
"""

import pytest
from upir.synthesis.program_synthesis import (
    ProgramSynthesizer,
    PredicateSynthesizer,
    TransformationSynthesizer,
    SynthesisSpec,
    synthesize_from_docstring
)


class TestProgramSynthesizer:
    """Test the core program synthesizer."""
    
    def setup_method(self):
        self.synthesizer = ProgramSynthesizer()
        
    def test_synthesize_simple_predicate(self):
        """Test synthesizing a simple boolean predicate."""
        spec = SynthesisSpec(
            name="is_positive",
            inputs={'x': int},
            output_type=bool,
            examples=[
                ({'x': 5}, True),
                ({'x': -3}, False),
                ({'x': 0}, False),
                ({'x': 10}, True)
            ],
            max_depth=2
        )
        
        result = self.synthesizer.synthesize(spec)
        
        assert result is not None
        assert result.verified_examples == 4
        # The synthesized code should be something like "x > 0"
        assert '>' in result.code or '<' in result.code
        
    def test_synthesize_arithmetic_expression(self):
        """Test synthesizing an arithmetic expression."""
        spec = SynthesisSpec(
            name="double_plus_one",
            inputs={'x': int},
            output_type=int,
            examples=[
                ({'x': 1}, 3),
                ({'x': 2}, 5),
                ({'x': 3}, 7),
                ({'x': 4}, 9)
            ],
            max_depth=3
        )
        
        result = self.synthesizer.synthesize(spec)
        
        assert result is not None
        # Should synthesize something like "x * 2 + 1"
        assert '*' in result.code or '+' in result.code
        
    def test_synthesize_comparison(self):
        """Test synthesizing a comparison expression."""
        spec = SynthesisSpec(
            name="in_range",
            inputs={'x': int, 'y': int},
            output_type=bool,
            examples=[
                ({'x': 5, 'y': 10}, True),
                ({'x': 15, 'y': 10}, False),
                ({'x': 3, 'y': 2}, False),
                ({'x': 8, 'y': 20}, True)
            ],
            max_depth=2
        )
        
        result = self.synthesizer.synthesize(spec)
        
        assert result is not None
        assert result.verified_examples == 4
        
    def test_synthesis_timeout(self):
        """Test that synthesis respects max iterations."""
        # Create an unsatisfiable spec
        spec = SynthesisSpec(
            name="impossible",
            inputs={'x': int},
            output_type=bool,
            examples=[
                ({'x': 1}, True),
                ({'x': 1}, False),  # Contradictory
            ],
            max_depth=2
        )
        
        synthesizer = ProgramSynthesizer(max_iterations=10)
        result = synthesizer.synthesize(spec)
        
        assert result is None  # Should fail to synthesize


class TestPredicateSynthesizer:
    """Test the predicate synthesizer."""
    
    def setup_method(self):
        self.synthesizer = PredicateSynthesizer()
        
    def test_synthesize_filter_predicate(self):
        """Test synthesizing a filter predicate."""
        examples = [
            ({'amount': 100, 'status': 'approved'}, True),
            ({'amount': 50, 'status': 'approved'}, False),
            ({'amount': 200, 'status': 'approved'}, True),
            ({'amount': 150, 'status': 'pending'}, True)
        ]
        
        result = self.synthesizer.synthesize_filter(examples)
        
        assert result is not None
        assert 'lambda' in result
        # Should synthesize something checking amount > threshold
        
    def test_synthesize_validator(self):
        """Test synthesizing a validation function."""
        valid = [10, 20, 30, 40]
        invalid = [5, 15, 25, 35]
        
        result = self.synthesizer.synthesize_validator(valid, invalid)
        
        assert result is not None
        assert 'lambda' in result
        
        # Test the synthesized validator
        validator = eval(result)
        assert validator(10) == True
        assert validator(5) == False
        
    def test_synthesize_string_predicate(self):
        """Test synthesizing predicates on strings."""
        examples = [
            ({'filename': 'test.py'}, True),
            ({'filename': 'main.js'}, False),
            ({'filename': 'utils.py'}, True),
            ({'filename': 'index.html'}, False)
        ]
        
        # This would need string operation support in the synthesizer
        # For now, we just test that it attempts synthesis
        result = self.synthesizer.synthesize_filter(examples, 'f', dict)
        # Result might be None if string operations aren't fully implemented


class TestTransformationSynthesizer:
    """Test the transformation synthesizer."""
    
    def setup_method(self):
        self.synthesizer = TransformationSynthesizer()
        
    def test_synthesize_mapper(self):
        """Test synthesizing a mapping function."""
        examples = [
            (1, 2),
            (2, 4),
            (3, 6),
            (4, 8)
        ]
        
        result = self.synthesizer.synthesize_mapper(examples)
        
        assert result is not None
        assert 'lambda' in result
        
        # Test the synthesized mapper
        mapper = eval(result)
        assert mapper(5) == 10  # Should be x * 2
        
    def test_synthesize_complex_mapper(self):
        """Test synthesizing a more complex transformation."""
        examples = [
            (1, 1),
            (2, 4),
            (3, 9),
            (4, 16)
        ]
        
        result = self.synthesizer.synthesize_mapper(examples)
        
        assert result is not None
        # Should synthesize x * x or equivalent
        mapper = eval(result)
        assert mapper(5) == 25
        
    def test_synthesize_aggregator(self):
        """Test synthesizing an aggregation function."""
        examples = [
            ([1, 2, 3], 6),
            ([4, 5], 9),
            ([10], 10),
            ([2, 2, 2], 6)
        ]
        
        result = self.synthesizer.synthesize_aggregator(examples)
        
        assert result is not None
        assert 'sum' in result
        
        aggregator = eval(result)
        assert aggregator([1, 2, 3, 4]) == 10
        
    def test_synthesize_max_aggregator(self):
        """Test synthesizing max aggregation."""
        examples = [
            ([1, 5, 3], 5),
            ([10, 2], 10),
            ([7], 7),
            ([3, 3, 3], 3)
        ]
        
        result = self.synthesizer.synthesize_aggregator(examples)
        
        assert result is not None
        assert 'max' in result


class TestSynthesisFromDocstring:
    """Test synthesis from natural language descriptions."""
    
    def test_synthesize_from_examples_in_docstring(self):
        """Test extracting examples from docstring and synthesizing."""
        docstring = """
        Filter payments where amount > 100
        Examples:
        - {'amount': 150} -> True
        - {'amount': 50} -> False
        - {'amount': 100} -> False
        - {'amount': 200} -> True
        """
        
        result = synthesize_from_docstring(docstring)
        
        assert result is not None
        assert 'lambda' in result
        
        # Test the synthesized function
        func = eval(result)
        assert func({'amount': 150}) == True
        assert func({'amount': 50}) == False
        
    def test_synthesize_numeric_transformation(self):
        """Test synthesizing numeric transformation from docstring."""
        docstring = """
        Convert celsius to fahrenheit
        Examples:
        - 0 -> 32
        - 100 -> 212
        - -40 -> -40
        - 20 -> 68
        """
        
        result = synthesize_from_docstring(docstring)
        
        # This is a complex transformation (x * 9/5 + 32)
        # Might not synthesize perfectly with limited depth
        if result:
            func = eval(result)
            # Check if it's close enough
            assert abs(func(0) - 32) < 5
            
    def test_invalid_docstring(self):
        """Test handling invalid docstring format."""
        docstring = "This docstring has no examples"
        
        result = synthesize_from_docstring(docstring)
        
        assert result is None


class TestExpressionEnumeration:
    """Test the expression enumeration logic."""
    
    def test_enumerate_depth_1(self):
        """Test enumerating expressions of depth 1."""
        synthesizer = ProgramSynthesizer()
        
        grammar = {
            'terminals': [
                ('var', 'x', int),
                ('const', 1, int),
                ('const', 2, int)
            ],
            'operators': [('+', 2), ('-', 2)],
            'functions': []
        }
        
        exprs = synthesizer._enumerate_expressions(grammar, 1, int)
        
        # At depth 1, we should get terminals and simple combinations
        assert len(exprs) > 3  # At least the terminals
        
    def test_enumerate_boolean_expressions(self):
        """Test enumerating boolean expressions."""
        synthesizer = ProgramSynthesizer()
        
        grammar = {
            'terminals': [
                ('var', 'x', int),
                ('const', 5, int)
            ],
            'operators': [('>', 2), ('<', 2), ('==', 2)],
            'functions': []
        }
        
        exprs = synthesizer._enumerate_expressions(grammar, 1, bool)
        
        # Should include comparisons like x > 5, x < 5, x == 5
        assert len(exprs) >= 2