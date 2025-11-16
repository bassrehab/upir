"""
Unit tests for program sketches with holes for CEGIS synthesis.

Tests verify:
- Hole: creation, validation, filling, Z3 variable conversion
- ProgramSketch: creation, hole management, instantiation
- Constraints: range, oneof, gt, lt, ne constraints
- Type handling: value, expression, predicate, function holes

Author: Subhadip Mitra
License: Apache 2.0
"""

import pytest

from upir.synthesis.sketch import Hole, ProgramSketch
from upir.verification.solver import is_z3_available

# Import Z3 if available for constraint tests
if is_z3_available():
    import z3


class TestHoleCreation:
    """Tests for Hole creation and validation."""

    def test_create_minimal_hole(self):
        """Test creating hole with minimal fields."""
        hole = Hole(
            id="h1",
            name="window_size",
            hole_type="value"
        )
        assert hole.id == "h1"
        assert hole.name == "window_size"
        assert hole.hole_type == "value"
        assert hole.constraints == []
        assert hole.possible_values is None
        assert hole.filled_value is None
        assert hole.location is None

    def test_create_complete_hole(self):
        """Test creating hole with all fields."""
        hole = Hole(
            id="h1",
            name="batch_size",
            hole_type="value",
            constraints=[("range", 1, 100)],
            possible_values=[10, 20, 50, 100],
            filled_value=20,
            location={"line": 42, "context": "GroupByKey"}
        )
        assert hole.id == "h1"
        assert hole.constraints == [("range", 1, 100)]
        assert hole.possible_values == [10, 20, 50, 100]
        assert hole.filled_value == 20
        assert hole.location == {"line": 42, "context": "GroupByKey"}

    def test_hole_types(self):
        """Test creating holes of different types."""
        types = ["value", "expression", "predicate", "function"]
        for hole_type in types:
            hole = Hole(id=f"h_{hole_type}", name=hole_type, hole_type=hole_type)
            assert hole.hole_type == hole_type

    def test_invalid_hole_type(self):
        """Test that invalid hole type raises error."""
        with pytest.raises(ValueError, match="Invalid hole_type"):
            Hole(id="h1", name="test", hole_type="invalid")

    def test_empty_id_rejected(self):
        """Test that empty ID is rejected."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            Hole(id="", name="test", hole_type="value")

    def test_empty_name_rejected(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Hole(id="h1", name="", hole_type="value")


class TestHoleFilling:
    """Tests for filling holes with values."""

    def test_is_filled_false_initially(self):
        """Test that holes are unfilled initially."""
        hole = Hole(id="h1", name="size", hole_type="value")
        assert hole.is_filled() is False

    def test_is_filled_true_after_filling(self):
        """Test that holes are filled after setting value."""
        hole = Hole(id="h1", name="size", hole_type="value")
        hole.filled_value = 42
        assert hole.is_filled() is True

    def test_fill_with_integer(self):
        """Test filling hole with integer value."""
        hole = Hole(id="h1", name="count", hole_type="value")
        hole.filled_value = 10
        assert hole.filled_value == 10

    def test_fill_with_float(self):
        """Test filling hole with float value."""
        hole = Hole(id="h1", name="threshold", hole_type="expression")
        hole.filled_value = 0.5
        assert hole.filled_value == 0.5

    def test_fill_with_boolean(self):
        """Test filling hole with boolean value."""
        hole = Hole(id="h1", name="enabled", hole_type="predicate")
        hole.filled_value = True
        assert hole.filled_value is True

    def test_fill_with_string(self):
        """Test filling hole with string value."""
        hole = Hole(id="h1", name="mode", hole_type="value")
        hole.filled_value = "batch"
        assert hole.filled_value == "batch"

    def test_fill_with_zero(self):
        """Test that filling with 0 is considered filled."""
        hole = Hole(id="h1", name="offset", hole_type="value")
        hole.filled_value = 0
        assert hole.is_filled() is True
        assert hole.filled_value == 0

    def test_fill_with_false(self):
        """Test that filling with False is considered filled."""
        hole = Hole(id="h1", name="flag", hole_type="predicate")
        hole.filled_value = False
        assert hole.is_filled() is True
        assert hole.filled_value is False


@pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
class TestHoleZ3Conversion:
    """Tests for converting holes to Z3 variables."""

    def test_value_hole_to_z3(self):
        """Test value hole converts to Z3 Int."""
        hole = Hole(id="h1", name="count", hole_type="value")
        var = hole.to_z3_var()
        assert isinstance(var, z3.ArithRef)
        assert var.decl().name() == "hole_h1"

    def test_expression_hole_to_z3(self):
        """Test expression hole converts to Z3 Real."""
        hole = Hole(id="h1", name="ratio", hole_type="expression")
        var = hole.to_z3_var()
        assert isinstance(var, z3.ArithRef)
        assert var.decl().name() == "hole_h1"

    def test_predicate_hole_to_z3(self):
        """Test predicate hole converts to Z3 Bool."""
        hole = Hole(id="h1", name="enabled", hole_type="predicate")
        var = hole.to_z3_var()
        assert isinstance(var, z3.BoolRef)
        assert var.decl().name() == "hole_h1"

    def test_function_hole_to_z3(self):
        """Test function hole returns None (special handling)."""
        hole = Hole(id="h1", name="transform", hole_type="function")
        var = hole.to_z3_var()
        assert var is None


@pytest.mark.skipif(not is_z3_available(), reason="Z3 not installed")
class TestHoleConstraints:
    """Tests for hole constraints and Z3 conversion."""

    def test_range_constraint(self):
        """Test range constraint converts to Z3."""
        hole = Hole(
            id="h1",
            name="size",
            hole_type="value",
            constraints=[("range", 1, 100)]
        )
        constraints = hole.get_constraints_as_z3()
        assert len(constraints) == 2  # >= 1 and <= 100

    def test_oneof_constraint(self):
        """Test oneof constraint converts to Z3."""
        hole = Hole(
            id="h1",
            name="mode",
            hole_type="value",
            constraints=[("oneof", [1, 2, 3, 4])]
        )
        constraints = hole.get_constraints_as_z3()
        assert len(constraints) == 1  # OR clause

    def test_gt_constraint(self):
        """Test greater than constraint."""
        hole = Hole(
            id="h1",
            name="threshold",
            hole_type="value",
            constraints=[("gt", 0)]
        )
        constraints = hole.get_constraints_as_z3()
        assert len(constraints) == 1

    def test_lt_constraint(self):
        """Test less than constraint."""
        hole = Hole(
            id="h1",
            name="limit",
            hole_type="value",
            constraints=[("lt", 1000)]
        )
        constraints = hole.get_constraints_as_z3()
        assert len(constraints) == 1

    def test_ne_constraint(self):
        """Test not equal constraint."""
        hole = Hole(
            id="h1",
            name="value",
            hole_type="value",
            constraints=[("ne", 0)]
        )
        constraints = hole.get_constraints_as_z3()
        assert len(constraints) == 1

    def test_multiple_constraints(self):
        """Test multiple constraints on same hole."""
        hole = Hole(
            id="h1",
            name="size",
            hole_type="value",
            constraints=[
                ("range", 1, 100),
                ("ne", 50)
            ]
        )
        constraints = hole.get_constraints_as_z3()
        assert len(constraints) == 3  # >= 1, <= 100, != 50

    def test_empty_constraints(self):
        """Test hole with no constraints."""
        hole = Hole(id="h1", name="value", hole_type="value")
        constraints = hole.get_constraints_as_z3()
        assert len(constraints) == 0

    def test_function_hole_no_constraints(self):
        """Test function hole has no Z3 constraints."""
        hole = Hole(
            id="h1",
            name="func",
            hole_type="function",
            constraints=[("range", 1, 100)]  # Ignored
        )
        constraints = hole.get_constraints_as_z3()
        assert len(constraints) == 0


class TestHoleStringRepresentation:
    """Tests for hole string representations."""

    def test_str_unfilled(self):
        """Test string representation of unfilled hole."""
        hole = Hole(id="h1", name="size", hole_type="value")
        s = str(hole)
        assert "size" in s
        assert "value" in s
        assert "UNFILLED" in s

    def test_str_filled(self):
        """Test string representation of filled hole."""
        hole = Hole(id="h1", name="size", hole_type="value")
        hole.filled_value = 42
        s = str(hole)
        assert "size" in s
        assert "FILLED" in s
        assert "42" in s

    def test_repr(self):
        """Test repr representation."""
        hole = Hole(id="h1", name="size", hole_type="value")
        r = repr(hole)
        assert "Hole" in r
        assert "h1" in r
        assert "size" in r
        assert "value" in r


class TestProgramSketchCreation:
    """Tests for ProgramSketch creation."""

    def test_create_minimal_sketch(self):
        """Test creating sketch with minimal fields."""
        sketch = ProgramSketch(template="x = __HOLE_h1__")
        assert sketch.template == "x = __HOLE_h1__"
        assert sketch.holes == []
        assert sketch.language == "python"
        assert sketch.framework == ""
        assert sketch.constraints == []

    def test_create_complete_sketch(self):
        """Test creating sketch with all fields."""
        template = "x = __HOLE_h1__ + __HOLE_h2__"
        holes = [
            Hole(id="h1", name="a", hole_type="value"),
            Hole(id="h2", name="b", hole_type="value")
        ]
        sketch = ProgramSketch(
            template=template,
            holes=holes,
            language="python",
            framework="Apache Beam",
            constraints=[("sum_constraint", "h1 + h2 <= 100")]
        )
        assert sketch.template == template
        assert len(sketch.holes) == 2
        assert sketch.language == "python"
        assert sketch.framework == "Apache Beam"
        assert len(sketch.constraints) == 1

    def test_empty_template_rejected(self):
        """Test that empty template is rejected."""
        with pytest.raises(ValueError, match="Template cannot be empty"):
            ProgramSketch(template="")

    def test_duplicate_hole_ids_rejected(self):
        """Test that duplicate hole IDs are rejected."""
        holes = [
            Hole(id="h1", name="a", hole_type="value"),
            Hole(id="h1", name="b", hole_type="value")  # Duplicate ID
        ]
        with pytest.raises(ValueError, match="Hole IDs must be unique"):
            ProgramSketch(template="test", holes=holes)


class TestProgramSketchHoleManagement:
    """Tests for managing holes in program sketch."""

    def test_get_unfilled_holes_all_unfilled(self):
        """Test getting unfilled holes when all are unfilled."""
        holes = [
            Hole(id="h1", name="a", hole_type="value"),
            Hole(id="h2", name="b", hole_type="value")
        ]
        sketch = ProgramSketch(template="test", holes=holes)
        unfilled = sketch.get_unfilled_holes()
        assert len(unfilled) == 2

    def test_get_unfilled_holes_partially_filled(self):
        """Test getting unfilled holes when some are filled."""
        holes = [
            Hole(id="h1", name="a", hole_type="value"),
            Hole(id="h2", name="b", hole_type="value")
        ]
        sketch = ProgramSketch(template="test", holes=holes)
        sketch.fill_hole("h1", 10)
        unfilled = sketch.get_unfilled_holes()
        assert len(unfilled) == 1
        assert unfilled[0].id == "h2"

    def test_get_unfilled_holes_all_filled(self):
        """Test getting unfilled holes when all are filled."""
        holes = [
            Hole(id="h1", name="a", hole_type="value"),
            Hole(id="h2", name="b", hole_type="value")
        ]
        sketch = ProgramSketch(template="test", holes=holes)
        sketch.fill_hole("h1", 10)
        sketch.fill_hole("h2", 20)
        unfilled = sketch.get_unfilled_holes()
        assert len(unfilled) == 0

    def test_fill_hole_success(self):
        """Test successfully filling a hole."""
        holes = [Hole(id="h1", name="size", hole_type="value")]
        sketch = ProgramSketch(template="test", holes=holes)
        result = sketch.fill_hole("h1", 42)
        assert result is True
        assert sketch.holes[0].filled_value == 42

    def test_fill_hole_not_found(self):
        """Test filling nonexistent hole returns False."""
        sketch = ProgramSketch(template="test")
        result = sketch.fill_hole("nonexistent", 42)
        assert result is False

    def test_fill_multiple_holes(self):
        """Test filling multiple holes."""
        holes = [
            Hole(id="h1", name="a", hole_type="value"),
            Hole(id="h2", name="b", hole_type="value"),
            Hole(id="h3", name="c", hole_type="value")
        ]
        sketch = ProgramSketch(template="test", holes=holes)
        sketch.fill_hole("h1", 10)
        sketch.fill_hole("h2", 20)
        sketch.fill_hole("h3", 30)
        assert all(h.is_filled() for h in sketch.holes)


class TestProgramSketchInstantiation:
    """Tests for instantiating program sketches."""

    def test_instantiate_single_hole(self):
        """Test instantiating sketch with single hole."""
        template = "x = __HOLE_h1__"
        sketch = ProgramSketch(
            template=template,
            holes=[Hole(id="h1", name="x", hole_type="value")]
        )
        sketch.fill_hole("h1", 42)
        code = sketch.instantiate()
        assert code == "x = 42"

    def test_instantiate_multiple_holes(self):
        """Test instantiating sketch with multiple holes."""
        template = "result = __HOLE_h1__ + __HOLE_h2__"
        sketch = ProgramSketch(
            template=template,
            holes=[
                Hole(id="h1", name="a", hole_type="value"),
                Hole(id="h2", name="b", hole_type="value")
            ]
        )
        sketch.fill_hole("h1", 10)
        sketch.fill_hole("h2", 20)
        code = sketch.instantiate()
        assert code == "result = 10 + 20"

    def test_instantiate_with_boolean(self):
        """Test instantiating hole with boolean value."""
        template = "enabled = __HOLE_h1__"
        sketch = ProgramSketch(
            template=template,
            holes=[Hole(id="h1", name="enabled", hole_type="predicate")]
        )
        sketch.fill_hole("h1", True)
        code = sketch.instantiate()
        assert code == "enabled = True"

    def test_instantiate_with_false(self):
        """Test instantiating hole with False value."""
        template = "debug = __HOLE_h1__"
        sketch = ProgramSketch(
            template=template,
            holes=[Hole(id="h1", name="debug", hole_type="predicate")]
        )
        sketch.fill_hole("h1", False)
        code = sketch.instantiate()
        assert code == "debug = False"

    def test_instantiate_with_string(self):
        """Test instantiating hole with string value."""
        template = "mode = __HOLE_h1__"
        sketch = ProgramSketch(
            template=template,
            holes=[Hole(id="h1", name="mode", hole_type="value")]
        )
        sketch.fill_hole("h1", "batch")
        code = sketch.instantiate()
        assert code == "mode = batch"

    def test_instantiate_with_quoted_string(self):
        """Test instantiating hole with already-quoted string."""
        template = "name = __HOLE_h1__"
        sketch = ProgramSketch(
            template=template,
            holes=[Hole(id="h1", name="name", hole_type="value")]
        )
        sketch.fill_hole("h1", '"my_name"')
        code = sketch.instantiate()
        assert code == 'name = "my_name"'

    def test_instantiate_unfilled_raises_error(self):
        """Test that instantiating with unfilled holes raises error."""
        sketch = ProgramSketch(
            template="x = __HOLE_h1__",
            holes=[Hole(id="h1", name="x", hole_type="value")]
        )
        with pytest.raises(ValueError, match="unfilled holes"):
            sketch.instantiate()

    def test_instantiate_apache_beam_example(self):
        """Test realistic Apache Beam example."""
        template = '''def process(data):
    windowed = data.window(
        window_size=__HOLE_h1__
    )
    return windowed.batch(
        batch_size=__HOLE_h2__
    )'''
        sketch = ProgramSketch(
            template=template,
            holes=[
                Hole(id="h1", name="window_size", hole_type="value"),
                Hole(id="h2", name="batch_size", hole_type="value")
            ],
            language="python",
            framework="Apache Beam"
        )
        sketch.fill_hole("h1", 60)
        sketch.fill_hole("h2", 100)
        code = sketch.instantiate()
        assert "window_size=60" in code
        assert "batch_size=100" in code


class TestProgramSketchSerialization:
    """Tests for serializing/deserializing program sketches."""

    def test_to_dict(self):
        """Test serializing sketch to dictionary."""
        sketch = ProgramSketch(
            template="x = __HOLE_h1__",
            holes=[Hole(id="h1", name="x", hole_type="value")],
            language="python"
        )
        d = sketch.to_dict()
        assert d["template"] == "x = __HOLE_h1__"
        assert len(d["holes"]) == 1
        assert d["language"] == "python"

    def test_from_dict(self):
        """Test deserializing sketch from dictionary."""
        data = {
            "template": "x = __HOLE_h1__",
            "holes": [{
                "id": "h1",
                "name": "x",
                "hole_type": "value",
                "constraints": [],
                "possible_values": None,
                "filled_value": None,
                "location": None
            }],
            "language": "python",
            "framework": "",
            "constraints": []
        }
        sketch = ProgramSketch.from_dict(data)
        assert sketch.template == "x = __HOLE_h1__"
        assert len(sketch.holes) == 1
        assert sketch.language == "python"

    def test_roundtrip_serialization(self):
        """Test that serialize->deserialize preserves sketch."""
        original = ProgramSketch(
            template="result = __HOLE_h1__ + __HOLE_h2__",
            holes=[
                Hole(id="h1", name="a", hole_type="value",
                     constraints=[("range", 1, 10)]),
                Hole(id="h2", name="b", hole_type="value",
                     constraints=[("range", 1, 10)])
            ],
            language="python",
            framework="Apache Beam"
        )
        data = original.to_dict()
        restored = ProgramSketch.from_dict(data)
        assert restored.template == original.template
        assert len(restored.holes) == len(original.holes)
        assert restored.language == original.language
        assert restored.framework == original.framework


class TestProgramSketchStringRepresentation:
    """Tests for sketch string representations."""

    def test_str_no_holes(self):
        """Test string representation with no holes."""
        sketch = ProgramSketch(template="test", language="python")
        s = str(sketch)
        assert "python" in s
        assert "0/0" in s

    def test_str_unfilled_holes(self):
        """Test string representation with unfilled holes."""
        sketch = ProgramSketch(
            template="test",
            holes=[
                Hole(id="h1", name="a", hole_type="value"),
                Hole(id="h2", name="b", hole_type="value")
            ],
            language="python"
        )
        s = str(sketch)
        assert "python" in s
        assert "0/2" in s

    def test_str_partially_filled(self):
        """Test string representation with partially filled holes."""
        sketch = ProgramSketch(
            template="test",
            holes=[
                Hole(id="h1", name="a", hole_type="value"),
                Hole(id="h2", name="b", hole_type="value")
            ]
        )
        sketch.fill_hole("h1", 10)
        s = str(sketch)
        assert "1/2" in s

    def test_repr(self):
        """Test repr representation."""
        sketch = ProgramSketch(
            template="test",
            holes=[Hole(id="h1", name="a", hole_type="value")],
            language="python"
        )
        r = repr(sketch)
        assert "ProgramSketch" in r
        assert "python" in r
        assert "holes=1" in r


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_hole_with_zero_value(self):
        """Test hole filled with zero is considered filled."""
        hole = Hole(id="h1", name="offset", hole_type="value")
        hole.filled_value = 0
        assert hole.is_filled() is True

    def test_hole_with_empty_string(self):
        """Test hole filled with empty string is considered filled."""
        hole = Hole(id="h1", name="prefix", hole_type="value")
        hole.filled_value = ""
        assert hole.is_filled() is True

    def test_sketch_with_same_hole_multiple_times(self):
        """Test sketch with same hole marker appearing multiple times."""
        template = "x = __HOLE_h1__ + __HOLE_h1__"
        sketch = ProgramSketch(
            template=template,
            holes=[Hole(id="h1", name="value", hole_type="value")]
        )
        sketch.fill_hole("h1", 5)
        code = sketch.instantiate()
        assert code == "x = 5 + 5"

    def test_complex_template(self):
        """Test complex template with multiple lines and holes."""
        template = '''
def configure_pipeline(pipeline):
    pipeline.set_parallelism(__HOLE_h1__)
    pipeline.set_batch_size(__HOLE_h2__)
    pipeline.set_timeout(__HOLE_h3__)
    return pipeline
'''
        sketch = ProgramSketch(
            template=template,
            holes=[
                Hole(id="h1", name="parallelism", hole_type="value"),
                Hole(id="h2", name="batch_size", hole_type="value"),
                Hole(id="h3", name="timeout", hole_type="value")
            ]
        )
        sketch.fill_hole("h1", 4)
        sketch.fill_hole("h2", 1000)
        sketch.fill_hole("h3", 30)
        code = sketch.instantiate()
        assert "set_parallelism(4)" in code
        assert "set_batch_size(1000)" in code
        assert "set_timeout(30)" in code
