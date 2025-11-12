"""
Program sketches with holes for CEGIS synthesis.

This module implements program sketches - partial programs with "holes" that
need to be filled in by the synthesis engine. Based on Counter-Example Guided
Inductive Synthesis (CEGIS), this enables automated program synthesis.

Implementation based on:
- CEGIS paper: Solar-Lezama et al. (2008)
  https://people.csail.mit.edu/asolar/papers/Solar-Lezama08.pdf
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Z3 Python API: https://z3prover.github.io/api/html/namespacez3py.html

Author: Subhadip Mitra
License: Apache 2.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from upir.verification.solver import is_z3_available

# Import Z3 if available
if is_z3_available():
    import z3
else:
    z3 = None


@dataclass
class Hole:
    """
    A hole in a program sketch that needs to be filled by synthesis.

    Per CEGIS methodology, holes represent unknown parts of a program that
    the synthesis engine must determine. Each hole has a type (value,
    expression, predicate, function) and constraints on valid values.

    Attributes:
        id: Unique identifier for this hole
        name: Descriptive name (e.g., "window_size", "batch_count")
        hole_type: Type of hole - "value", "expression", "predicate", "function"
        constraints: List of constraints on valid values
                    Format: [("range", min, max), ("oneof", [v1, v2, v3]), ...]
        possible_values: Optional explicit list of possible values
        filled_value: The value this hole has been filled with (if any)
        location: Optional location info (line number, context)

    Example:
        >>> # Integer value hole with range constraint
        >>> hole = Hole(
        ...     id="h1",
        ...     name="window_size",
        ...     hole_type="value",
        ...     constraints=[("range", 1, 100)],
        ...     location={"line": 42, "context": "GroupByKey"}
        ... )
        >>> hole.is_filled()
        False
        >>> hole.filled_value = 10
        >>> hole.is_filled()
        True

    References:
    - CEGIS: Holes are unknowns to be synthesized
    - TD Commons: Program sketch representation
    """

    id: str
    name: str
    hole_type: str  # "value", "expression", "predicate", "function"
    constraints: List[Any] = field(default_factory=list)
    possible_values: Optional[List[Any]] = None
    filled_value: Optional[Any] = None
    location: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate hole configuration."""
        valid_types = ["value", "expression", "predicate", "function"]
        if self.hole_type not in valid_types:
            raise ValueError(
                f"Invalid hole_type: {self.hole_type}. "
                f"Must be one of {valid_types}"
            )

        if not self.id:
            raise ValueError("Hole id cannot be empty")

        if not self.name:
            raise ValueError("Hole name cannot be empty")

    def is_filled(self) -> bool:
        """
        Check if this hole has been filled with a value.

        Returns:
            True if hole has a filled_value, False otherwise

        Example:
            >>> hole = Hole(id="h1", name="size", hole_type="value")
            >>> hole.is_filled()
            False
            >>> hole.filled_value = 42
            >>> hole.is_filled()
            True
        """
        return self.filled_value is not None

    def to_z3_var(self) -> Any:
        """
        Convert hole to Z3 variable for SMT-based synthesis.

        Creates appropriate Z3 variable based on hole type:
        - "value": Integer variable (z3.Int)
        - "expression": Real variable (z3.Real)
        - "predicate": Boolean variable (z3.Bool)
        - "function": Returns None (requires special handling)

        Returns:
            Z3 variable or None for function holes

        Raises:
            RuntimeError: If Z3 is not available

        Example:
            >>> hole = Hole(id="h1", name="count", hole_type="value")
            >>> var = hole.to_z3_var()
            >>> # var is z3.Int("hole_h1")

        References:
        - Z3 API: Variable creation
        - CEGIS: Encoding unknowns as SMT variables
        """
        if not is_z3_available():
            raise RuntimeError(
                "Z3 solver is not available. "
                "Install with: pip install z3-solver"
            )

        var_name = f"hole_{self.id}"

        if self.hole_type == "value":
            # Integer value hole
            return z3.Int(var_name)

        elif self.hole_type == "expression":
            # Real-valued expression hole
            return z3.Real(var_name)

        elif self.hole_type == "predicate":
            # Boolean predicate hole
            return z3.Bool(var_name)

        elif self.hole_type == "function":
            # Function holes require special handling
            # Cannot be represented as simple Z3 variable
            return None

        else:
            # Should not reach here due to __post_init__ validation
            raise ValueError(f"Unknown hole type: {self.hole_type}")

    def get_constraints_as_z3(self) -> List[Any]:
        """
        Convert hole constraints to Z3 constraints.

        Translates constraint specifications into Z3 constraint expressions
        that can be added to the solver.

        Returns:
            List of Z3 constraint expressions

        Example:
            >>> hole = Hole(
            ...     id="h1",
            ...     name="size",
            ...     hole_type="value",
            ...     constraints=[("range", 1, 100)]
            ... )
            >>> z3_constraints = hole.get_constraints_as_z3()
            >>> # z3_constraints[0] is (hole_h1 >= 1)
            >>> # z3_constraints[1] is (hole_h1 <= 100)

        References:
        - Z3: Constraint construction
        - CEGIS: Encoding hole constraints
        """
        if not is_z3_available():
            raise RuntimeError("Z3 solver is not available")

        var = self.to_z3_var()
        if var is None:
            return []  # Function holes have no direct constraints

        z3_constraints = []

        for constraint in self.constraints:
            if not constraint:
                continue

            constraint_type = constraint[0] if isinstance(constraint, tuple) else None

            if constraint_type == "range" and len(constraint) >= 3:
                # Range constraint: min <= var <= max
                min_val = constraint[1]
                max_val = constraint[2]
                z3_constraints.append(var >= min_val)
                z3_constraints.append(var <= max_val)

            elif constraint_type == "oneof" and len(constraint) >= 2:
                # OneOf constraint: var in {v1, v2, v3, ...}
                values = constraint[1]
                if values:
                    # Create disjunction: var == v1 OR var == v2 OR ...
                    or_clauses = [var == v for v in values]
                    z3_constraints.append(z3.Or(or_clauses))

            elif constraint_type == "gt" and len(constraint) >= 2:
                # Greater than constraint
                value = constraint[1]
                z3_constraints.append(var > value)

            elif constraint_type == "lt" and len(constraint) >= 2:
                # Less than constraint
                value = constraint[1]
                z3_constraints.append(var < value)

            elif constraint_type == "ne" and len(constraint) >= 2:
                # Not equal constraint
                value = constraint[1]
                z3_constraints.append(var != value)

        return z3_constraints

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "FILLED" if self.is_filled() else "UNFILLED"
        value_str = f"={self.filled_value}" if self.is_filled() else ""
        return f"Hole({self.name}:{self.hole_type}, {status}{value_str})"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"Hole(id='{self.id}', name='{self.name}', "
            f"type='{self.hole_type}', filled={self.is_filled()})"
        )


@dataclass
class ProgramSketch:
    """
    A program sketch - partial program with holes to be filled by synthesis.

    Per CEGIS methodology, a sketch is a template program with "holes"
    representing unknown parts. The synthesis engine fills these holes
    to produce a complete program that satisfies the specification.

    Template uses __HOLE_{id}__ markers that get replaced during instantiation.

    Attributes:
        template: Code template with __HOLE_{id}__ markers
        holes: List of holes to fill
        language: Programming language (e.g., "python", "java")
        framework: Framework/platform (e.g., "Apache Beam", "Spark")
        constraints: Global constraints across multiple holes

    Example:
        >>> template = '''
        ... def process(data):
        ...     windowed = data.window(
        ...         window_size=__HOLE_h1__
        ...     )
        ...     return windowed.batch(
        ...         batch_size=__HOLE_h2__
        ...     )
        ... '''
        >>> sketch = ProgramSketch(
        ...     template=template,
        ...     holes=[
        ...         Hole(id="h1", name="window_size", hole_type="value",
        ...              constraints=[("range", 1, 60)]),
        ...         Hole(id="h2", name="batch_size", hole_type="value",
        ...              constraints=[("range", 1, 1000)])
        ...     ],
        ...     language="python",
        ...     framework="Apache Beam"
        ... )
        >>> unfilled = sketch.get_unfilled_holes()
        >>> len(unfilled)
        2
        >>> sketch.fill_hole("h1", 10)
        True
        >>> sketch.fill_hole("h2", 100)
        True
        >>> code = sketch.instantiate()
        >>> "window_size=10" in code
        True

    References:
    - CEGIS: Program sketch definition
    - TD Commons: Sketch-based synthesis
    """

    template: str
    holes: List[Hole] = field(default_factory=list)
    language: str = "python"
    framework: str = ""
    constraints: List[Any] = field(default_factory=list)

    def __post_init__(self):
        """Validate program sketch configuration."""
        if not self.template:
            raise ValueError("Template cannot be empty")

        # Verify all hole IDs are unique
        hole_ids = [h.id for h in self.holes]
        if len(hole_ids) != len(set(hole_ids)):
            raise ValueError("Hole IDs must be unique")

    def get_unfilled_holes(self) -> List[Hole]:
        """
        Get list of holes that have not been filled yet.

        Returns:
            List of unfilled Hole objects

        Example:
            >>> sketch = ProgramSketch(
            ...     template="x = __HOLE_h1__ + __HOLE_h2__",
            ...     holes=[
            ...         Hole(id="h1", name="a", hole_type="value"),
            ...         Hole(id="h2", name="b", hole_type="value")
            ...     ]
            ... )
            >>> len(sketch.get_unfilled_holes())
            2
            >>> sketch.fill_hole("h1", 10)
            True
            >>> len(sketch.get_unfilled_holes())
            1
        """
        return [hole for hole in self.holes if not hole.is_filled()]

    def fill_hole(self, hole_id: str, value: Any) -> bool:
        """
        Fill a hole with a specific value.

        Args:
            hole_id: ID of the hole to fill
            value: Value to fill the hole with

        Returns:
            True if hole was filled successfully, False if hole not found

        Example:
            >>> sketch = ProgramSketch(
            ...     template="size = __HOLE_h1__",
            ...     holes=[Hole(id="h1", name="size", hole_type="value")]
            ... )
            >>> sketch.fill_hole("h1", 42)
            True
            >>> sketch.fill_hole("nonexistent", 10)
            False
        """
        for hole in self.holes:
            if hole.id == hole_id:
                hole.filled_value = value
                return True
        return False

    def instantiate(self) -> str:
        """
        Instantiate the sketch by replacing all holes with their filled values.

        Replaces all __HOLE_{id}__ markers in the template with the
        corresponding filled values. Handles type conversion (bool to
        "True"/"False", etc.).

        Returns:
            Complete code with all holes filled

        Raises:
            ValueError: If any holes are unfilled

        Example:
            >>> template = "result = __HOLE_h1__ + __HOLE_h2__"
            >>> sketch = ProgramSketch(
            ...     template=template,
            ...     holes=[
            ...         Hole(id="h1", name="a", hole_type="value"),
            ...         Hole(id="h2", name="b", hole_type="value")
            ...     ]
            ... )
            >>> sketch.fill_hole("h1", 10)
            True
            >>> sketch.fill_hole("h2", 20)
            True
            >>> code = sketch.instantiate()
            >>> code
            'result = 10 + 20'

        References:
        - CEGIS: Sketch instantiation
        - String replacement for code generation
        """
        unfilled = self.get_unfilled_holes()
        if unfilled:
            unfilled_names = [h.name for h in unfilled]
            raise ValueError(
                f"Cannot instantiate sketch with unfilled holes: {unfilled_names}"
            )

        # Start with template
        code = self.template

        # Replace each hole marker with its filled value
        for hole in self.holes:
            marker = f"__HOLE_{hole.id}__"
            value = hole.filled_value

            # Convert value to string representation
            if isinstance(value, bool):
                # Python boolean: True/False (capitalized)
                value_str = str(value)
            elif isinstance(value, str):
                # String: keep quotes if needed
                # Check if it looks like a variable name or literal
                if value.startswith('"') or value.startswith("'"):
                    value_str = value  # Already quoted
                else:
                    # Could be variable name - don't quote
                    value_str = value
            elif value is None:
                value_str = "None"
            else:
                # Numeric or other types
                value_str = str(value)

            # Replace all occurrences of this marker
            code = code.replace(marker, value_str)

        return code

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize program sketch to dictionary.

        Returns:
            Dictionary with all sketch fields

        Example:
            >>> sketch = ProgramSketch(
            ...     template="x = __HOLE_h1__",
            ...     holes=[Hole(id="h1", name="x", hole_type="value")],
            ...     language="python"
            ... )
            >>> d = sketch.to_dict()
            >>> d["language"]
            'python'
        """
        return {
            "template": self.template,
            "holes": [
                {
                    "id": h.id,
                    "name": h.name,
                    "hole_type": h.hole_type,
                    "constraints": h.constraints,
                    "possible_values": h.possible_values,
                    "filled_value": h.filled_value,
                    "location": h.location
                }
                for h in self.holes
            ],
            "language": self.language,
            "framework": self.framework,
            "constraints": self.constraints
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgramSketch":
        """
        Deserialize program sketch from dictionary.

        Args:
            data: Dictionary containing sketch fields

        Returns:
            ProgramSketch instance

        Example:
            >>> data = {
            ...     "template": "x = __HOLE_h1__",
            ...     "holes": [{
            ...         "id": "h1",
            ...         "name": "x",
            ...         "hole_type": "value",
            ...         "constraints": [],
            ...         "possible_values": None,
            ...         "filled_value": None,
            ...         "location": None
            ...     }],
            ...     "language": "python",
            ...     "framework": "",
            ...     "constraints": []
            ... }
            >>> sketch = ProgramSketch.from_dict(data)
            >>> sketch.language
            'python'
        """
        holes = [
            Hole(
                id=h["id"],
                name=h["name"],
                hole_type=h["hole_type"],
                constraints=h.get("constraints", []),
                possible_values=h.get("possible_values"),
                filled_value=h.get("filled_value"),
                location=h.get("location")
            )
            for h in data.get("holes", [])
        ]

        return cls(
            template=data["template"],
            holes=holes,
            language=data.get("language", "python"),
            framework=data.get("framework", ""),
            constraints=data.get("constraints", [])
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        filled_count = len(self.holes) - len(self.get_unfilled_holes())
        total_count = len(self.holes)
        return (
            f"ProgramSketch({self.language}, "
            f"{filled_count}/{total_count} holes filled)"
        )

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"ProgramSketch(language='{self.language}', "
            f"holes={len(self.holes)}, "
            f"unfilled={len(self.get_unfilled_holes())})"
        )
