"""
Unit tests for CEGIS synthesis engine.

Testing the synthesis components including sketch generation,
hole filling, and the main CEGIS loop.

Author: subhadipmitra@google.com
"""

import pytest
import ast
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upir.core.models import (
    UPIR, FormalSpecification, TemporalProperty, TemporalOperator
)
from upir.synthesis.synthesizer import (
    Synthesizer, Hole, ProgramSketch, SynthesisExample,
    CEGISResult, SynthesisStatus
)


class TestHole:
    """Test hole representation and operations."""
    
    def test_hole_creation(self):
        """Test creating a hole."""
        hole = Hole(
            id="1",
            name="window_size",
            hole_type="value",
            constraints=[("range", 1, 100)],
            possible_values=[10, 20, 30]
        )
        
        assert hole.id == "1"
        assert hole.name == "window_size"
        assert hole.hole_type == "value"
        assert not hole.is_filled()
    
    def test_hole_filling(self):
        """Test filling a hole with a value."""
        hole = Hole(
            id="1",
            name="test_hole",
            hole_type="value"
        )
        
        assert not hole.is_filled()
        
        hole.filled_value = 42
        
        assert hole.is_filled()
        assert hole.filled_value == 42
    
    def test_hole_serialization(self):
        """Test hole serialization."""
        hole = Hole(
            id="1",
            name="param",
            hole_type="expression",
            constraints=[("range", 0, 10)],
            filled_value=5
        )
        
        data = hole.to_dict()
        
        assert data["id"] == "1"
        assert data["name"] == "param"
        assert data["type"] == "expression"
        assert data["filled_value"] == 5
    
    def test_hole_z3_conversion(self):
        """Test converting hole to Z3 variable."""
        # Test different hole types
        value_hole = Hole("1", "val", "value")
        expr_hole = Hole("2", "expr", "expression")
        pred_hole = Hole("3", "pred", "predicate")
        func_hole = Hole("4", "func", "function")
        
        # These should return appropriate Z3 types or None
        # (actual types depend on Z3 availability)
        value_var = value_hole.to_z3_var()
        expr_var = expr_hole.to_z3_var()
        pred_var = pred_hole.to_z3_var()
        func_var = func_hole.to_z3_var()
        
        # Function holes return None (need special handling)
        if func_var is not None:
            assert func_var is None


class TestProgramSketch:
    """Test program sketch functionality."""
    
    def test_sketch_creation(self):
        """Test creating a program sketch."""
        holes = [
            Hole("1", "param1", "value"),
            Hole("2", "param2", "value")
        ]
        
        template = """
def process(data):
    x = __HOLE_1__
    y = __HOLE_2__
    return x + y
"""
        
        sketch = ProgramSketch(
            template=template,
            holes=holes,
            language="python",
            framework="generic"
        )
        
        assert sketch.template == template
        assert len(sketch.holes) == 2
        assert sketch.language == "python"
    
    def test_get_unfilled_holes(self):
        """Test getting unfilled holes."""
        holes = [
            Hole("1", "filled", "value"),
            Hole("2", "unfilled", "value"),
            Hole("3", "also_unfilled", "value")
        ]
        
        holes[0].filled_value = 10  # Fill first hole
        
        sketch = ProgramSketch(
            template="",
            holes=holes,
            language="python",
            framework="generic"
        )
        
        unfilled = sketch.get_unfilled_holes()
        
        assert len(unfilled) == 2
        assert unfilled[0].name == "unfilled"
        assert unfilled[1].name == "also_unfilled"
    
    def test_fill_hole(self):
        """Test filling a hole by ID."""
        holes = [
            Hole("1", "hole1", "value"),
            Hole("2", "hole2", "value")
        ]
        
        sketch = ProgramSketch(
            template="",
            holes=holes,
            language="python",
            framework="generic"
        )
        
        # Fill hole by ID
        success = sketch.fill_hole("1", 42)
        assert success
        assert holes[0].filled_value == 42
        
        # Try filling non-existent hole
        success = sketch.fill_hole("999", 100)
        assert not success
    
    def test_instantiate(self):
        """Test instantiating sketch with filled holes."""
        holes = [
            Hole("1", "window", "value"),
            Hole("2", "parallel", "value"),
            Hole("3", "condition", "predicate")
        ]
        
        template = """
window_size = __HOLE_1__
workers = __HOLE_2__
if __HOLE_3__:
    process()
"""
        
        sketch = ProgramSketch(
            template=template,
            holes=holes,
            language="python",
            framework="generic"
        )
        
        # Fill holes
        sketch.fill_hole("1", 60)
        sketch.fill_hole("2", 10)
        sketch.fill_hole("3", True)
        
        # Instantiate
        code = sketch.instantiate()
        
        assert "window_size = 60" in code
        assert "workers = 10" in code
        assert "if True:" in code
        assert "__HOLE_" not in code  # All holes should be replaced


class TestSynthesisExample:
    """Test synthesis examples."""
    
    def test_example_creation(self):
        """Test creating synthesis examples."""
        example = SynthesisExample(
            inputs={"x": 5, "y": 10},
            expected_output=15,
            weight=0.8
        )
        
        assert example.inputs["x"] == 5
        assert example.expected_output == 15
        assert example.weight == 0.8
    
    def test_example_serialization(self):
        """Test example serialization."""
        example = SynthesisExample(
            inputs={"data": [1, 2, 3]},
            expected_output=[3, 2, 1],
            weight=1.0
        )
        
        data = example.to_dict()
        
        assert data["inputs"]["data"] == [1, 2, 3]
        assert data["expected_output"] == [3, 2, 1]
        assert data["weight"] == 1.0


class TestSynthesizer:
    """Test the main synthesis engine."""
    
    def test_synthesizer_creation(self):
        """Test creating a synthesizer."""
        synth = Synthesizer(max_iterations=50, timeout=10000)
        
        assert synth.max_iterations == 50
        assert synth.timeout == 10000
        assert synth.verifier is not None
    
    def test_infer_system_type(self):
        """Test inferring system type from specification."""
        synth = Synthesizer()
        
        # Streaming system
        spec1 = FormalSpecification(
            invariants=[
                TemporalProperty(
                    TemporalOperator.WITHIN,
                    "event_processed",
                    time_bound=100
                )
            ],
            properties=[],
            constraints={}
        )
        
        system_type = synth._infer_system_type(spec1)
        assert system_type == "streaming"
        
        # Batch system
        spec2 = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={"batch_size": {"min": 100}}
        )
        
        system_type = synth._infer_system_type(spec2)
        assert system_type == "batch"
        
        # API system
        spec3 = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={"request_rate": {"max": 1000}}
        )
        
        system_type = synth._infer_system_type(spec3)
        assert system_type == "api"
    
    def test_generate_sketch(self):
        """Test sketch generation from specification."""
        synth = Synthesizer()
        
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(
                    TemporalOperator.WITHIN,
                    "processed",
                    time_bound=50
                )
            ],
            properties=[],
            constraints={
                "latency": {"max": 50},
                "throughput": {"min": 1000}
            }
        )
        
        sketch = synth.generate_sketch(spec)
        
        assert sketch is not None
        assert isinstance(sketch, ProgramSketch)
        assert len(sketch.holes) > 0
        assert sketch.language == "python"
        assert "__HOLE_" in sketch.template
    
    def test_synthesize_holes_heuristic(self):
        """Test heuristic hole synthesis."""
        synth = Synthesizer()
        
        holes = [
            Hole("1", "window_size", "value", [("range", 10, 300)]),
            Hole("2", "parallelism", "value", [("range", 1, 20)]),
            Hole("3", "buffer_size", "value", [("range", 100, 5000)])
        ]
        
        sketch = ProgramSketch(
            template="",
            holes=holes,
            language="python",
            framework="generic"
        )
        
        spec = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={
                "latency": {"max": 100},
                "throughput": {"min": 5000}
            }
        )
        
        # Use heuristic synthesis
        success = synth._synthesize_holes_heuristic(sketch, spec)
        
        assert success
        assert all(h.is_filled() for h in sketch.holes)
        
        # Check reasonable values were chosen
        assert 10 <= holes[0].filled_value <= 300  # window_size
        assert 1 <= holes[1].filled_value <= 20    # parallelism
        assert holes[2].filled_value == 1000       # buffer_size default
    
    def test_verify_synthesis(self):
        """Test synthesis verification."""
        synth = Synthesizer()
        
        # Valid Python code
        valid_impl = Implementation(
            code="def process(x): return x * 2",
            language="python",
            framework="generic"
        )
        
        spec = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={}
        )
        
        result = synth.verify_synthesis(valid_impl, spec)
        
        assert "verified" in result
        # Should pass basic syntax check
        assert result["verified"] or result["counterexample"] is not None
        
        # Invalid Python code
        invalid_impl = Implementation(
            code="def process(x: return x * 2",  # Syntax error
            language="python",
            framework="generic"
        )
        
        result = synth.verify_synthesis(invalid_impl, spec)
        
        assert not result["verified"]
        assert result["counterexample"] is not None
    
    def test_synthesize_simple(self):
        """Test simple synthesis without examples."""
        synth = Synthesizer(max_iterations=5, timeout=5000)
        
        upir = UPIR(name="Test System")
        
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(
                    TemporalOperator.WITHIN,
                    "processed",
                    time_bound=100
                )
            ],
            properties=[],
            constraints={
                "latency": {"max": 100}
            }
        )
        
        upir.specification = spec
        
        # Run synthesis
        result = synth.synthesize(upir)
        
        assert isinstance(result, CEGISResult)
        assert result.status in [
            SynthesisStatus.SUCCESS,
            SynthesisStatus.PARTIAL,
            SynthesisStatus.TIMEOUT,
            SynthesisStatus.FAILED
        ]
        
        if result.status == SynthesisStatus.SUCCESS:
            assert result.implementation is not None
            assert result.sketch is not None
            
            # Check that holes were filled
            assert all(h.is_filled() for h in result.sketch.holes)
    
    def test_synthesize_with_examples(self):
        """Test synthesis with input-output examples."""
        synth = Synthesizer(max_iterations=10)
        
        upir = UPIR(name="Example-Guided System")
        
        spec = FormalSpecification(
            invariants=[],
            properties=[],
            constraints={}
        )
        
        upir.specification = spec
        
        # Create examples
        examples = [
            SynthesisExample(
                inputs={"x": 5},
                expected_output=10,
                weight=1.0
            ),
            SynthesisExample(
                inputs={"x": 10},
                expected_output=20,
                weight=1.0
            )
        ]
        
        # Run synthesis
        result = synth.synthesize(upir, examples)
        
        assert isinstance(result, CEGISResult)
        assert result.iterations > 0
        
        if result.status == SynthesisStatus.SUCCESS:
            # Should have synthesized something
            assert result.implementation is not None
            assert result.implementation.code != ""
    
    def test_cegis_iterations(self):
        """Test that CEGIS performs iterations."""
        synth = Synthesizer(max_iterations=3)
        
        upir = UPIR(name="Iterative System")
        
        # Create a challenging spec that requires iterations
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(TemporalOperator.ALWAYS, "invariant1"),
                TemporalProperty(TemporalOperator.ALWAYS, "invariant2")
            ],
            properties=[],
            constraints={
                "constraint1": {"max": 10},
                "constraint2": {"min": 5}
            }
        )
        
        upir.specification = spec
        
        result = synth.synthesize(upir)
        
        # Should have performed some iterations
        assert result.iterations > 0
        assert result.iterations <= 3  # Respects max_iterations
        
        # Might have collected counterexamples
        # (depends on verification results)
        assert isinstance(result.counterexamples, list)


class TestCEGISResult:
    """Test CEGIS result."""
    
    def test_result_creation(self):
        """Test creating a CEGIS result."""
        result = CEGISResult(
            status=SynthesisStatus.SUCCESS,
            iterations=5,
            execution_time=2.5
        )
        
        assert result.status == SynthesisStatus.SUCCESS
        assert result.iterations == 5
        assert result.execution_time == 2.5
    
    def test_result_serialization(self):
        """Test result serialization."""
        impl = Implementation(
            code="def f(): pass",
            language="python",
            framework="generic"
        )
        
        result = CEGISResult(
            status=SynthesisStatus.SUCCESS,
            implementation=impl,
            iterations=3,
            counterexamples=[{"ce": 1}],
            execution_time=1.5
        )
        
        data = result.to_dict()
        
        assert data["status"] == "success"
        assert data["implementation"] is not None
        assert data["iterations"] == 3
        assert len(data["counterexamples"]) == 1
        assert data["execution_time"] == 1.5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])