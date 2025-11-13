"""
Unit tests for CEGIS (Counterexample-Guided Inductive Synthesis).

Tests verify:
- SynthesisStatus enum
- SynthesisExample dataclass
- CEGISResult dataclass
- Synthesizer: sketch generation, hole synthesis, verification, CEGIS loop
- Timeout and iteration limit handling

Author: Subhadip Mitra
License: Apache 2.0
"""

import pytest
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.core.upir import UPIR
from upir.synthesis.cegis import (
    CEGISResult,
    SynthesisExample,
    SynthesisStatus,
    Synthesizer,
)
from upir.synthesis.sketch import Hole, ProgramSketch
from upir.verification.solver import is_z3_available


class TestSynthesisStatus:
    """Tests for SynthesisStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert SynthesisStatus.SUCCESS.value == "SUCCESS"
        assert SynthesisStatus.FAILED.value == "FAILED"
        assert SynthesisStatus.TIMEOUT.value == "TIMEOUT"
        assert SynthesisStatus.PARTIAL.value == "PARTIAL"
        assert SynthesisStatus.INVALID_SPEC.value == "INVALID_SPEC"

    def test_status_from_string(self):
        """Test creating status from string value."""
        assert SynthesisStatus("SUCCESS") == SynthesisStatus.SUCCESS
        assert SynthesisStatus("FAILED") == SynthesisStatus.FAILED
        assert SynthesisStatus("TIMEOUT") == SynthesisStatus.TIMEOUT
        assert SynthesisStatus("PARTIAL") == SynthesisStatus.PARTIAL
        assert SynthesisStatus("INVALID_SPEC") == SynthesisStatus.INVALID_SPEC


class TestSynthesisExample:
    """Tests for SynthesisExample dataclass."""

    def test_create_minimal_example(self):
        """Test creating example with minimal fields."""
        ex = SynthesisExample(
            inputs={"x": 5},
            expected_output=10
        )
        assert ex.inputs == {"x": 5}
        assert ex.expected_output == 10
        assert ex.weight == 1.0  # Default

    def test_create_complete_example(self):
        """Test creating example with all fields."""
        ex = SynthesisExample(
            inputs={"x": 2, "y": 3},
            expected_output=5,
            weight=2.0
        )
        assert ex.inputs == {"x": 2, "y": 3}
        assert ex.expected_output == 5
        assert ex.weight == 2.0

    def test_example_with_complex_inputs(self):
        """Test example with complex input types."""
        ex = SynthesisExample(
            inputs={"data": [1, 2, 3], "config": {"mode": "batch"}},
            expected_output={"result": [2, 4, 6]}
        )
        assert ex.inputs["data"] == [1, 2, 3]
        assert ex.inputs["config"] == {"mode": "batch"}
        assert ex.expected_output == {"result": [2, 4, 6]}


class TestCEGISResult:
    """Tests for CEGISResult dataclass."""

    def test_create_minimal_result(self):
        """Test creating result with minimal fields."""
        result = CEGISResult(status=SynthesisStatus.SUCCESS)
        assert result.status == SynthesisStatus.SUCCESS
        assert result.implementation is None
        assert result.sketch is None
        assert result.iterations == 0
        assert result.counterexamples == []
        assert result.execution_time == 0.0

    def test_create_complete_result(self):
        """Test creating result with all fields."""
        sketch = ProgramSketch(
            template="x = __HOLE_h1__",
            holes=[Hole("h1", "x", "value")]
        )
        result = CEGISResult(
            status=SynthesisStatus.SUCCESS,
            implementation="x = 42",
            sketch=sketch,
            iterations=5,
            counterexamples=[{"x": 0}],
            execution_time=1.23
        )
        assert result.status == SynthesisStatus.SUCCESS
        assert result.implementation == "x = 42"
        assert result.sketch == sketch
        assert result.iterations == 5
        assert len(result.counterexamples) == 1
        assert result.execution_time == 1.23

    def test_result_str(self):
        """Test string representation."""
        result = CEGISResult(
            status=SynthesisStatus.SUCCESS,
            implementation="def f(x): return x * 2",
            iterations=3,
            execution_time=0.5
        )
        s = str(result)
        assert "SUCCESS" in s
        assert "iterations=3" in s
        assert "0.50s" in s

    def test_result_str_long_implementation(self):
        """Test string representation with long implementation."""
        long_code = "def f(x):\n    " + "x " * 50  # Very long code
        result = CEGISResult(
            status=SynthesisStatus.SUCCESS,
            implementation=long_code,
            iterations=1,
            execution_time=0.1
        )
        s = str(result)
        assert "SUCCESS" in s
        assert "..." in s  # Truncated


@pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
class TestSynthesizerCreation:
    """Tests for Synthesizer creation."""

    def test_create_default_synthesizer(self):
        """Test creating synthesizer with default parameters."""
        synth = Synthesizer()
        assert synth.max_iterations == 100
        assert synth.timeout == 60000

    def test_create_custom_synthesizer(self):
        """Test creating synthesizer with custom parameters."""
        synth = Synthesizer(max_iterations=50, timeout=30000)
        assert synth.max_iterations == 50
        assert synth.timeout == 30000

    def test_synthesizer_str(self):
        """Test string representation."""
        synth = Synthesizer(max_iterations=25, timeout=15000)
        s = str(synth)
        assert "Synthesizer" in s
        assert "25" in s
        assert "15000" in s


@pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
class TestGenerateSketch:
    """Tests for sketch generation."""

    def test_generate_sketch_from_spec(self):
        """Test generating sketch from specification."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="result_positive"
                )
            ]
        )
        synth = Synthesizer()
        sketch = synth.generate_sketch(spec)

        assert isinstance(sketch, ProgramSketch)
        assert len(sketch.holes) > 0
        assert sketch.language == "python"

    def test_generate_sketch_creates_holes(self):
        """Test that generated sketch has holes to fill."""
        spec = FormalSpecification()
        synth = Synthesizer()
        sketch = synth.generate_sketch(spec)

        assert len(sketch.holes) > 0
        for hole in sketch.holes:
            assert isinstance(hole, Hole)
            assert not hole.is_filled()


@pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
class TestSynthesizeHoles:
    """Tests for hole synthesis."""

    def test_synthesize_holes_success(self):
        """Test successfully synthesizing hole values."""
        sketch = ProgramSketch(
            template="x = __HOLE_h1__",
            holes=[
                Hole(
                    id="h1",
                    name="x",
                    hole_type="value",
                    constraints=[("range", 1, 10)]
                )
            ]
        )
        spec = FormalSpecification()
        examples = [SynthesisExample({"x": 1}, 1)]

        synth = Synthesizer()
        success = synth.synthesize_holes(sketch, spec, examples, [])

        assert success is True
        assert sketch.holes[0].is_filled()
        assert 1 <= sketch.holes[0].filled_value <= 10

    def test_synthesize_multiple_holes(self):
        """Test synthesizing multiple holes."""
        sketch = ProgramSketch(
            template="result = __HOLE_h1__ + __HOLE_h2__",
            holes=[
                Hole("h1", "a", "value", [("range", 0, 50)]),
                Hole("h2", "b", "value", [("range", 0, 50)])
            ]
        )
        spec = FormalSpecification()
        examples = []

        synth = Synthesizer()
        success = synth.synthesize_holes(sketch, spec, examples, [])

        assert success is True
        assert all(h.is_filled() for h in sketch.holes)

    def test_synthesize_expression_hole(self):
        """Test synthesizing expression (real) hole."""
        sketch = ProgramSketch(
            template="threshold = __HOLE_h1__",
            holes=[
                Hole("h1", "threshold", "expression", [("range", 0, 1)])
            ]
        )
        spec = FormalSpecification()
        examples = []

        synth = Synthesizer()
        success = synth.synthesize_holes(sketch, spec, examples, [])

        assert success is True
        assert sketch.holes[0].is_filled()

    def test_synthesize_predicate_hole(self):
        """Test synthesizing predicate (boolean) hole."""
        sketch = ProgramSketch(
            template="enabled = __HOLE_h1__",
            holes=[Hole("h1", "enabled", "predicate")]
        )
        spec = FormalSpecification()
        examples = []

        synth = Synthesizer()
        success = synth.synthesize_holes(sketch, spec, examples, [])

        assert success is True
        assert sketch.holes[0].is_filled()
        assert isinstance(sketch.holes[0].filled_value, bool)

    def test_synthesize_with_counterexamples(self):
        """Test synthesis with counterexamples (should still succeed)."""
        sketch = ProgramSketch(
            template="x = __HOLE_h1__",
            holes=[Hole("h1", "x", "value", [("range", 1, 100)])]
        )
        spec = FormalSpecification()
        examples = []
        counterexamples = [{"x": 50}, {"x": 75}]

        synth = Synthesizer()
        success = synth.synthesize_holes(sketch, spec, examples, counterexamples)

        # Should still find a solution
        assert success is True


@pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
class TestVerifySynthesis:
    """Tests for synthesis verification."""

    def test_verify_synthesis_success(self):
        """Test verification of synthesized code."""
        synth = Synthesizer()
        spec = FormalSpecification()

        result = synth.verify_synthesis("def f(x): return x", spec)

        assert isinstance(result, dict)
        assert "verified" in result
        # Current implementation always returns True (placeholder)
        assert result["verified"] is True

    def test_verify_synthesis_returns_counterexample(self):
        """Test verification result includes counterexample field."""
        synth = Synthesizer()
        spec = FormalSpecification()

        result = synth.verify_synthesis("code", spec)

        assert "counterexample" in result


@pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
class TestCEGISLoop:
    """Tests for main CEGIS synthesis loop."""

    def test_synthesize_invalid_spec(self):
        """Test synthesis with invalid UPIR (no specification)."""
        upir = UPIR(id="test", name="Test", description="Test")
        examples = []

        synth = Synthesizer()
        result = synth.synthesize(upir, examples)

        assert result.status == SynthesisStatus.INVALID_SPEC
        assert result.execution_time > 0

    def test_synthesize_success(self):
        """Test successful synthesis."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="positive"
                )
            ]
        )
        upir = UPIR(
            id="test",
            name="Test",
            description="Test",
            specification=spec
        )
        examples = [
            SynthesisExample({"x": 1}, 1),
            SynthesisExample({"x": 2}, 4)
        ]

        synth = Synthesizer(max_iterations=10, timeout=5000)
        result = synth.synthesize(upir, examples)

        # Current implementation always succeeds (placeholder verification)
        assert result.status == SynthesisStatus.SUCCESS
        assert result.implementation is not None
        assert result.sketch is not None
        assert result.iterations > 0
        assert result.execution_time > 0

    def test_synthesize_creates_sketch(self):
        """Test that synthesis creates a sketch."""
        spec = FormalSpecification()
        upir = UPIR(
            id="test",
            name="Test",
            description="Test",
            specification=spec
        )
        examples = []

        synth = Synthesizer(max_iterations=5)
        result = synth.synthesize(upir, examples)

        assert result.sketch is not None
        assert isinstance(result.sketch, ProgramSketch)

    def test_synthesize_with_examples(self):
        """Test synthesis with input/output examples."""
        spec = FormalSpecification()
        upir = UPIR(
            id="test",
            name="Test",
            description="Test",
            specification=spec
        )
        examples = [
            SynthesisExample({"x": 0}, 0, weight=1.0),
            SynthesisExample({"x": 1}, 1, weight=1.0),
            SynthesisExample({"x": 2}, 4, weight=1.0)
        ]

        synth = Synthesizer(max_iterations=10)
        result = synth.synthesize(upir, examples)

        assert result.status in [
            SynthesisStatus.SUCCESS,
            SynthesisStatus.PARTIAL,
            SynthesisStatus.TIMEOUT
        ]

    def test_synthesize_iteration_count(self):
        """Test that synthesis tracks iteration count."""
        spec = FormalSpecification()
        upir = UPIR(
            id="test",
            name="Test",
            description="Test",
            specification=spec
        )
        examples = []

        synth = Synthesizer(max_iterations=5, timeout=5000)
        result = synth.synthesize(upir, examples)

        assert result.iterations > 0
        assert result.iterations <= 5

    def test_synthesize_timeout(self):
        """Test synthesis timeout handling."""
        spec = FormalSpecification()
        upir = UPIR(
            id="test",
            name="Test",
            description="Test",
            specification=spec
        )
        examples = []

        # Very short timeout
        synth = Synthesizer(max_iterations=1000, timeout=1)
        result = synth.synthesize(upir, examples)

        # Should timeout or succeed very quickly
        assert result.status in [
            SynthesisStatus.SUCCESS,
            SynthesisStatus.TIMEOUT
        ]
        assert result.execution_time < 1.0  # Less than 1 second

    def test_synthesize_max_iterations(self):
        """Test synthesis respects max iterations."""
        spec = FormalSpecification()
        upir = UPIR(
            id="test",
            name="Test",
            description="Test",
            specification=spec
        )
        examples = []

        synth = Synthesizer(max_iterations=3, timeout=60000)
        result = synth.synthesize(upir, examples)

        # Should complete within 3 iterations
        assert result.iterations <= 3


class TestSynthesizerWithoutZ3:
    """Tests for Synthesizer when Z3 is not available."""

    def test_synthesizer_requires_z3(self):
        """Test that Synthesizer raises error if Z3 not available."""
        if is_z3_available():
            pytest.skip("Z3 is available, cannot test unavailable case")

        with pytest.raises(RuntimeError, match="Z3 solver is not available"):
            Synthesizer()


@pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
class TestPatternInference:
    """Tests for pattern-specific sketch generation."""

    def test_infer_streaming_from_keywords(self):
        """Test streaming inference from keywords."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="process_stream_events"
                )
            ]
        )
        synth = Synthesizer()
        system_type = synth._infer_system_type(spec)
        assert system_type == "streaming"

    def test_infer_streaming_from_latency(self):
        """Test streaming inference from low latency requirement."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.WITHIN,
                    predicate="respond",
                    time_bound=500  # 500ms - very low latency
                )
            ]
        )
        synth = Synthesizer()
        system_type = synth._infer_system_type(spec)
        assert system_type == "streaming"

    def test_infer_batch_from_keywords(self):
        """Test batch inference from keywords."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="batch_job_completes"
                )
            ]
        )
        synth = Synthesizer()
        system_type = synth._infer_system_type(spec)
        assert system_type == "batch"

    def test_infer_api_from_keywords(self):
        """Test API inference from keywords."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="api_request_handled"
                )
            ]
        )
        synth = Synthesizer()
        system_type = synth._infer_system_type(spec)
        assert system_type == "api"

    def test_infer_generic_default(self):
        """Test generic inference as default."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="something_unknown"
                )
            ]
        )
        synth = Synthesizer()
        system_type = synth._infer_system_type(spec)
        assert system_type == "generic"

    def test_generate_streaming_sketch(self):
        """Test streaming sketch generation."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="stream_processing"
                )
            ]
        )
        synth = Synthesizer()
        sketch = synth.generate_sketch(spec)

        # Should generate streaming sketch
        assert isinstance(sketch, ProgramSketch)
        assert "apache_beam" in sketch.template.lower()
        assert "streaming" in sketch.template.lower()

        # Should have 3 holes: window_size, parallelism, buffer_size
        assert len(sketch.holes) == 3
        hole_names = {h.name for h in sketch.holes}
        assert "window_size" in hole_names
        assert "parallelism" in hole_names
        assert "buffer_size" in hole_names

        # Check constraints
        for hole in sketch.holes:
            if hole.name == "window_size":
                assert hole.hole_type == "value"
                assert len(hole.constraints) > 0
                # Should have range constraint 1-3600
                assert any(c[0] == "range" and c[1] == 1 and c[2] == 3600
                          for c in hole.constraints if isinstance(c, tuple))
            elif hole.name == "parallelism":
                assert hole.hole_type == "value"
                # Should have range constraint 1-100
                assert any(c[0] == "range" and c[1] == 1 and c[2] == 100
                          for c in hole.constraints if isinstance(c, tuple))
            elif hole.name == "buffer_size":
                assert hole.hole_type == "value"
                # Should have range constraint 100-10000
                assert any(c[0] == "range" and c[1] == 100 and c[2] == 10000
                          for c in hole.constraints if isinstance(c, tuple))

    def test_generate_batch_sketch(self):
        """Test batch sketch generation."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="batch_processing"
                )
            ]
        )
        synth = Synthesizer()
        sketch = synth.generate_sketch(spec)

        # Should generate batch sketch
        assert isinstance(sketch, ProgramSketch)
        assert "apache_beam" in sketch.template.lower()

        # Should have 2 holes: batch_size, parallelism
        assert len(sketch.holes) == 2
        hole_names = {h.name for h in sketch.holes}
        assert "batch_size" in hole_names
        assert "parallelism" in hole_names

        # Check constraints
        for hole in sketch.holes:
            if hole.name == "batch_size":
                assert hole.hole_type == "value"
                # Should have range constraint 100-10000
                assert any(c[0] == "range" and c[1] == 100 and c[2] == 10000
                          for c in hole.constraints if isinstance(c, tuple))
            elif hole.name == "parallelism":
                assert hole.hole_type == "value"
                # Should have range constraint 1-50
                assert any(c[0] == "range" and c[1] == 1 and c[2] == 50
                          for c in hole.constraints if isinstance(c, tuple))

    def test_generate_api_sketch(self):
        """Test API sketch generation."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="api_endpoint_available"
                )
            ]
        )
        synth = Synthesizer()
        sketch = synth.generate_sketch(spec)

        # Should generate API sketch
        assert isinstance(sketch, ProgramSketch)
        assert "fastapi" in sketch.template.lower() or "api" in sketch.template.lower()

        # Should have 2 holes: max_connections, timeout
        assert len(sketch.holes) == 2
        hole_names = {h.name for h in sketch.holes}
        assert "max_connections" in hole_names
        assert "timeout" in hole_names

        # Check constraints
        for hole in sketch.holes:
            if hole.name == "max_connections":
                assert hole.hole_type == "value"
                # Should have range constraint 10-1000
                assert any(c[0] == "range" and c[1] == 10 and c[2] == 1000
                          for c in hole.constraints if isinstance(c, tuple))
            elif hole.name == "timeout":
                assert hole.hole_type == "value"
                # Should have range constraint 100-30000
                assert any(c[0] == "range" and c[1] == 100 and c[2] == 30000
                          for c in hole.constraints if isinstance(c, tuple))

    def test_generate_generic_sketch(self):
        """Test generic sketch generation."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="custom_requirement"
                )
            ]
        )
        synth = Synthesizer()
        sketch = synth.generate_sketch(spec)

        # Should generate generic sketch
        assert isinstance(sketch, ProgramSketch)
        assert "def synthesized_function" in sketch.template

        # Should have 2 holes: param1, param2
        assert len(sketch.holes) == 2
        hole_names = {h.name for h in sketch.holes}
        assert "param1" in hole_names
        assert "param2" in hole_names

        # Check constraints
        for hole in sketch.holes:
            assert hole.hole_type == "value"
            # Should have range constraint 0-100
            assert any(c[0] == "range" and c[1] == 0 and c[2] == 100
                      for c in hole.constraints if isinstance(c, tuple))

    def test_streaming_with_pubsub_keyword(self):
        """Test streaming inference with 'pubsub' keyword."""
        spec = FormalSpecification(
            properties=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="pubsub_messages_delivered"
                )
            ]
        )
        synth = Synthesizer()
        system_type = synth._infer_system_type(spec)
        assert system_type == "streaming"

    def test_batch_with_etl_keyword(self):
        """Test batch inference with 'etl' keyword."""
        spec = FormalSpecification(
            invariants=[
                TemporalProperty(
                    operator=TemporalOperator.ALWAYS,
                    predicate="etl_pipeline_completes"
                )
            ]
        )
        synth = Synthesizer()
        system_type = synth._infer_system_type(spec)
        assert system_type == "batch"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
    def test_synthesize_empty_examples(self):
        """Test synthesis with no examples."""
        spec = FormalSpecification()
        upir = UPIR(
            id="test",
            name="Test",
            description="Test",
            specification=spec
        )
        examples = []

        synth = Synthesizer(max_iterations=5)
        result = synth.synthesize(upir, examples)

        # Should still succeed (examples are optional)
        assert result.status in [
            SynthesisStatus.SUCCESS,
            SynthesisStatus.PARTIAL
        ]

    @pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
    def test_example_with_zero_weight(self):
        """Test example with zero weight."""
        ex = SynthesisExample(
            inputs={"x": 1},
            expected_output=1,
            weight=0.0
        )
        assert ex.weight == 0.0

    @pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
    def test_result_with_no_counterexamples(self):
        """Test result with empty counterexamples list."""
        result = CEGISResult(
            status=SynthesisStatus.SUCCESS,
            counterexamples=[]
        )
        assert len(result.counterexamples) == 0

    @pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
    def test_result_with_multiple_counterexamples(self):
        """Test result tracking multiple counterexamples."""
        result = CEGISResult(
            status=SynthesisStatus.PARTIAL,
            counterexamples=[
                {"x": 0},
                {"x": 5},
                {"x": 10}
            ]
        )
        assert len(result.counterexamples) == 3
