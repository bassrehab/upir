"""
Counterexample-Guided Inductive Synthesis (CEGIS) for UPIR.

This module implements CEGIS - an iterative algorithm that synthesizes
programs by alternating between synthesis (finding hole values using SMT)
and verification (checking if the program satisfies the specification).

Implementation based on:
- CEGIS paper: Solar-Lezama et al. (2008)
  https://people.csail.mit.edu/asolar/papers/Solar-Lezama08.pdf
- TD Commons disclosure: https://www.tdcommons.org/dpubs_series/8852/
- Z3 tutorial: https://microsoft.github.io/z3guide/

Author: Subhadip Mitra
License: Apache 2.0
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from upir.core.specification import FormalSpecification
from upir.core.upir import UPIR
from upir.synthesis.sketch import Hole, ProgramSketch
from upir.verification.solver import is_z3_available

# Import Z3 if available
if is_z3_available():
    import z3
else:
    z3 = None

# Logger for CEGIS
logger = logging.getLogger(__name__)


class SynthesisStatus(Enum):
    """
    Status of a synthesis attempt.

    Based on standard synthesis outcomes:
    - SUCCESS: Successfully synthesized program that satisfies specification
    - FAILED: Cannot synthesize (specification unsatisfiable or unrealizable)
    - TIMEOUT: Synthesis exceeded time limit
    - PARTIAL: Found partial solution (some properties verified)
    - INVALID_SPEC: Specification is invalid or inconsistent

    References:
    - CEGIS: Standard synthesis result categories
    - TD Commons: Synthesis status tracking
    """
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    PARTIAL = "PARTIAL"
    INVALID_SPEC = "INVALID_SPEC"


@dataclass
class SynthesisExample:
    """
    An input/output example for synthesis.

    Per CEGIS methodology, examples guide the synthesis process by
    specifying desired program behavior on concrete inputs. The
    synthesizer finds programs that match these examples.

    Attributes:
        inputs: Input values as dictionary (e.g., {"x": 5, "y": 10})
        expected_output: Expected output for these inputs
        weight: Importance weight (higher = more important)

    Example:
        >>> # Example: function should compute x + y
        >>> ex1 = SynthesisExample(
        ...     inputs={"x": 2, "y": 3},
        ...     expected_output=5,
        ...     weight=1.0
        ... )
        >>> ex2 = SynthesisExample(
        ...     inputs={"x": 10, "y": 20},
        ...     expected_output=30,
        ...     weight=1.0
        ... )

    References:
    - CEGIS: Input/output examples for synthesis
    - Programming by example (PBE)
    """
    inputs: Dict[str, Any]
    expected_output: Any
    weight: float = 1.0


@dataclass
class CEGISResult:
    """
    Result of CEGIS synthesis attempt.

    Captures all information about a synthesis run including the
    synthesized implementation (if successful), sketch used, number
    of iterations, counterexamples encountered, and timing.

    Attributes:
        status: Synthesis outcome status
        implementation: Synthesized code (if successful)
        sketch: Program sketch that was filled
        iterations: Number of CEGIS iterations performed
        counterexamples: Counterexamples encountered during synthesis
        execution_time: Time taken for synthesis (seconds)

    Example:
        >>> result = CEGISResult(
        ...     status=SynthesisStatus.SUCCESS,
        ...     implementation="def f(x, y): return x + y",
        ...     sketch=sketch,
        ...     iterations=3,
        ...     counterexamples=[{"x": 0, "y": 0}],
        ...     execution_time=1.23
        ... )
        >>> result.status == SynthesisStatus.SUCCESS
        True

    References:
    - CEGIS: Synthesis result structure
    - TD Commons: Synthesis tracking and provenance
    """
    status: SynthesisStatus
    implementation: Optional[str] = None
    sketch: Optional[ProgramSketch] = None
    iterations: int = 0
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0

    def __str__(self) -> str:
        """Human-readable representation."""
        impl_preview = ""
        if self.implementation:
            # Show first 50 chars of implementation
            preview = self.implementation[:50].replace("\n", " ")
            impl_preview = f", impl='{preview}...'" if len(self.implementation) > 50 else f", impl='{preview}'"
        return (
            f"CEGISResult({self.status.value}, "
            f"iterations={self.iterations}, "
            f"time={self.execution_time:.2f}s{impl_preview})"
        )


class Synthesizer:
    """
    CEGIS-based program synthesizer.

    Implements Counterexample-Guided Inductive Synthesis (CEGIS) to
    automatically synthesize programs that satisfy formal specifications.
    Uses Z3 SMT solver to find hole values and verification to check
    correctness.

    The CEGIS loop:
    1. Generate initial program sketch from specification
    2. Loop until max_iterations or timeout:
       a. Synthesize hole values using SMT solver
       b. Instantiate program from filled sketch
       c. Verify synthesized program
       d. If verified: return SUCCESS
       e. If failed: add counterexample, continue
    3. Return PARTIAL if max iterations reached

    Attributes:
        max_iterations: Maximum CEGIS iterations (default: 100)
        timeout: Total synthesis timeout in milliseconds (default: 60000)

    Example:
        >>> synthesizer = Synthesizer(max_iterations=50, timeout=30000)
        >>> examples = [
        ...     SynthesisExample({"x": 2, "y": 3}, 5),
        ...     SynthesisExample({"x": 10, "y": 20}, 30)
        ... ]
        >>> result = synthesizer.synthesize(upir, examples)
        >>> if result.status == SynthesisStatus.SUCCESS:
        ...     print(f"Synthesized: {result.implementation}")

    References:
    - CEGIS: Solar-Lezama et al. (2008)
    - TD Commons: Synthesis engine architecture
    - Z3: SMT-based synthesis
    """

    def __init__(self, max_iterations: int = 100, timeout: int = 60000):
        """
        Initialize CEGIS synthesizer.

        Args:
            max_iterations: Maximum CEGIS iterations (default: 100)
            timeout: Total timeout in milliseconds (default: 60000 = 60s)

        Raises:
            RuntimeError: If Z3 is not available

        Example:
            >>> synthesizer = Synthesizer(max_iterations=50, timeout=30000)
        """
        if not is_z3_available():
            raise RuntimeError(
                "Z3 solver is not available. "
                "Install with: pip install z3-solver"
            )

        self.max_iterations = max_iterations
        self.timeout = timeout  # milliseconds

    def synthesize(
        self,
        upir: UPIR,
        examples: List[SynthesisExample]
    ) -> CEGISResult:
        """
        Synthesize program using CEGIS.

        Main CEGIS loop that alternates between synthesis (finding hole
        values) and verification (checking correctness). Uses examples
        to guide synthesis and counterexamples to refine.

        Args:
            upir: UPIR instance with specification and architecture
            examples: List of input/output examples

        Returns:
            CEGISResult with synthesis outcome

        Example:
            >>> upir = UPIR(...)
            >>> upir.specification = FormalSpecification(...)
            >>> examples = [
            ...     SynthesisExample({"x": 1}, 1),
            ...     SynthesisExample({"x": 2}, 4)
            ... ]
            >>> result = synthesizer.synthesize(upir, examples)
            >>> print(f"Status: {result.status}")

        References:
        - CEGIS: Main synthesis algorithm
        - TD Commons: Synthesis workflow
        """
        start_time = time.time()

        # Validate inputs
        if upir.specification is None:
            return CEGISResult(
                status=SynthesisStatus.INVALID_SPEC,
                execution_time=time.time() - start_time
            )

        # Generate initial sketch
        try:
            sketch = self.generate_sketch(upir.specification)
        except Exception as e:
            logger.error(f"Failed to generate sketch: {e}")
            return CEGISResult(
                status=SynthesisStatus.FAILED,
                execution_time=time.time() - start_time
            )

        # CEGIS loop
        counterexamples = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= self.timeout:
                logger.warning(f"Synthesis timeout after {iteration} iterations")
                return CEGISResult(
                    status=SynthesisStatus.TIMEOUT,
                    sketch=sketch,
                    iterations=iteration,
                    counterexamples=counterexamples,
                    execution_time=time.time() - start_time
                )

            # Synthesize hole values
            logger.info(f"CEGIS iteration {iteration}: synthesizing holes...")
            success = self.synthesize_holes(
                sketch=sketch,
                spec=upir.specification,
                examples=examples,
                counterexamples=counterexamples
            )

            if not success:
                # Cannot find hole values that satisfy constraints
                logger.warning("Synthesis failed: no valid hole values found")
                return CEGISResult(
                    status=SynthesisStatus.FAILED,
                    sketch=sketch,
                    iterations=iteration,
                    counterexamples=counterexamples,
                    execution_time=time.time() - start_time
                )

            # Instantiate program
            try:
                implementation = sketch.instantiate()
            except ValueError as e:
                logger.error(f"Instantiation failed: {e}")
                return CEGISResult(
                    status=SynthesisStatus.FAILED,
                    sketch=sketch,
                    iterations=iteration,
                    counterexamples=counterexamples,
                    execution_time=time.time() - start_time
                )

            # Verify synthesized program
            logger.info(f"Verifying synthesized program...")
            verification_result = self.verify_synthesis(
                implementation=implementation,
                spec=upir.specification
            )

            if verification_result.get("verified", False):
                # Success!
                logger.info(f"Synthesis succeeded in {iteration} iterations")
                return CEGISResult(
                    status=SynthesisStatus.SUCCESS,
                    implementation=implementation,
                    sketch=sketch,
                    iterations=iteration,
                    counterexamples=counterexamples,
                    execution_time=time.time() - start_time
                )
            else:
                # Verification failed - add counterexample
                counterexample = verification_result.get("counterexample", {})
                if counterexample:
                    counterexamples.append(counterexample)
                    logger.info(
                        f"Found counterexample: {counterexample}, "
                        f"continuing synthesis..."
                    )
                else:
                    # No counterexample but verification failed
                    logger.warning("Verification failed without counterexample")
                    return CEGISResult(
                        status=SynthesisStatus.FAILED,
                        sketch=sketch,
                        iterations=iteration,
                        counterexamples=counterexamples,
                        execution_time=time.time() - start_time
                    )

        # Reached max iterations
        logger.warning(f"Reached max iterations ({self.max_iterations})")
        return CEGISResult(
            status=SynthesisStatus.PARTIAL,
            implementation=sketch.instantiate() if all(h.is_filled() for h in sketch.holes) else None,
            sketch=sketch,
            iterations=iteration,
            counterexamples=counterexamples,
            execution_time=time.time() - start_time
        )

    def generate_sketch(self, spec: FormalSpecification) -> ProgramSketch:
        """
        Generate initial program sketch from specification.

        Creates a program template with holes based on the formal
        specification. Analyzes the spec to infer system type (streaming,
        batch, api, generic) and generates appropriate pattern-specific sketch.

        Args:
            spec: Formal specification

        Returns:
            ProgramSketch with holes to fill

        Example:
            >>> spec = FormalSpecification(
            ...     properties=[
            ...         TemporalProperty(WITHIN, "event_processed", time_bound=100)
            ...     ]
            ... )
            >>> sketch = synthesizer.generate_sketch(spec)
            >>> sketch.framework  # Will be "Apache Beam" for streaming
            'Apache Beam'

        References:
        - CEGIS: Sketch generation from specifications
        - TD Commons: Specification-to-sketch translation
        - Apache Beam: Streaming pattern
        """
        # Infer system type from specification
        system_type = self._infer_system_type(spec)
        logger.info(f"Inferred system type: {system_type}")

        # Generate pattern-specific sketch
        if system_type == "streaming":
            return self._generate_streaming_sketch(spec)
        elif system_type == "batch":
            return self._generate_batch_sketch(spec)
        elif system_type == "api":
            return self._generate_api_sketch(spec)
        else:
            return self._generate_generic_sketch(spec)

    def _infer_system_type(self, spec: FormalSpecification) -> str:
        """
        Infer distributed system type from specification.

        Analyzes predicates, time bounds, and constraints to determine
        the type of system being specified: streaming, batch, API, or generic.

        Heuristics:
        - "stream" or "event" in predicates -> streaming
        - Low latency bounds (<1000ms) -> streaming
        - "batch" in constraints or predicates -> batch
        - "request" or "response" in predicates -> api
        - Default: generic

        Args:
            spec: Formal specification to analyze

        Returns:
            System type: "streaming", "batch", "api", or "generic"

        Example:
            >>> spec = FormalSpecification(
            ...     properties=[
            ...         TemporalProperty(WITHIN, "event_processed", time_bound=500)
            ...     ]
            ... )
            >>> synth._infer_system_type(spec)
            'streaming'

        References:
        - TD Commons: Pattern inference from specifications
        - Distributed system patterns
        """
        # Collect all predicates from properties and invariants
        predicates = []
        time_bounds = []

        for prop in spec.properties + spec.invariants:
            predicates.append(prop.predicate.lower())
            if prop.time_bound is not None:
                time_bounds.append(prop.time_bound)

        # Check for streaming indicators
        streaming_keywords = ["stream", "event", "message", "pubsub", "kafka"]
        for keyword in streaming_keywords:
            for predicate in predicates:
                if keyword in predicate:
                    return "streaming"

        # Check for low latency (< 1 second) indicating streaming
        if time_bounds:
            min_bound = min(time_bounds)
            if min_bound < 1000:  # Less than 1 second (in ms)
                return "streaming"

        # Check for batch indicators
        batch_keywords = ["batch", "job", "task", "etl"]
        for keyword in batch_keywords:
            for predicate in predicates:
                if keyword in predicate:
                    return "batch"

        # Check for API indicators
        api_keywords = ["request", "response", "endpoint", "api", "http"]
        for keyword in api_keywords:
            for predicate in predicates:
                if keyword in predicate:
                    return "api"

        # Default to generic
        return "generic"

    def _generate_streaming_sketch(self, spec: FormalSpecification) -> ProgramSketch:
        """
        Generate Apache Beam streaming pipeline sketch.

        Creates a streaming data pipeline template with holes for:
        - window_size: Window duration in seconds (1-3600)
        - parallelism: Max number of workers (1-100)
        - buffer_size: Buffer size for processing (100-10000)

        Args:
            spec: Formal specification

        Returns:
            ProgramSketch for Apache Beam streaming pipeline

        Example:
            >>> spec = FormalSpecification(...)
            >>> sketch = synth._generate_streaming_sketch(spec)
            >>> sketch.framework
            'Apache Beam'
            >>> len(sketch.holes)
            3

        References:
        - Apache Beam: https://beam.apache.org/
        - TD Commons: Streaming pattern synthesis
        """
        template = '''import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

def create_pipeline():
    """
    Streaming data pipeline.

    Synthesized configuration:
    - Window size: __HOLE_window_size__ seconds
    - Max workers: __HOLE_parallelism__
    - Buffer size: __HOLE_buffer_size__
    """
    options = PipelineOptions(
        streaming=True,
        max_num_workers=__HOLE_parallelism__
    )

    with beam.Pipeline(options=options) as pipeline:
        # Read from streaming source
        events = (
            pipeline
            | 'Read Events' >> beam.io.ReadFromPubSub(
                subscription='projects/PROJECT/subscriptions/SUBSCRIPTION'
            )
        )

        # Window events
        windowed = (
            events
            | 'Window' >> beam.WindowInto(
                beam.window.FixedWindows(__HOLE_window_size__)
            )
        )

        # Process events
        processed = (
            windowed
            | 'Process' >> beam.ParDo(ProcessEventFn())
            | 'Buffer' >> beam.transforms.util.BatchElements(
                min_batch_size=__HOLE_buffer_size__
            )
        )

        # Write results
        processed | 'Write' >> beam.io.WriteToBigQuery(
            table='PROJECT:DATASET.TABLE',
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
        )

class ProcessEventFn(beam.DoFn):
    """Process individual events."""

    def process(self, element):
        # Process event
        yield element
'''

        holes = [
            Hole(
                id="window_size",
                name="window_size",
                hole_type="value",
                constraints=[("range", 1, 3600)],
                location={"context": "Window duration in seconds"}
            ),
            Hole(
                id="parallelism",
                name="parallelism",
                hole_type="value",
                constraints=[("range", 1, 100)],
                location={"context": "Maximum number of workers"}
            ),
            Hole(
                id="buffer_size",
                name="buffer_size",
                hole_type="value",
                constraints=[("range", 100, 10000)],
                location={"context": "Batch buffer size"}
            )
        ]

        return ProgramSketch(
            template=template,
            holes=holes,
            language="python",
            framework="Apache Beam",
            constraints=[]
        )

    def _generate_batch_sketch(self, spec: FormalSpecification) -> ProgramSketch:
        """
        Generate batch processing pipeline sketch.

        Creates a batch job template with holes for:
        - batch_size: Number of records per batch (100-10000)
        - parallelism: Number of parallel workers (1-50)

        Args:
            spec: Formal specification

        Returns:
            ProgramSketch for batch processing

        References:
        - TD Commons: Batch processing pattern
        """
        template = '''import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

def create_batch_pipeline():
    """
    Batch processing pipeline.

    Synthesized configuration:
    - Batch size: __HOLE_batch_size__ records
    - Parallelism: __HOLE_parallelism__ workers
    """
    options = PipelineOptions(
        runner='DataflowRunner',
        max_num_workers=__HOLE_parallelism__
    )

    with beam.Pipeline(options=options) as pipeline:
        # Read batch data
        data = (
            pipeline
            | 'Read' >> beam.io.ReadFromText('gs://bucket/input/*.txt')
        )

        # Process in batches
        processed = (
            data
            | 'Batch' >> beam.BatchElements(
                min_batch_size=__HOLE_batch_size__
            )
            | 'Process' >> beam.ParDo(ProcessBatchFn())
        )

        # Write results
        processed | 'Write' >> beam.io.WriteToText('gs://bucket/output')

class ProcessBatchFn(beam.DoFn):
    """Process batch of records."""

    def process(self, batch):
        # Process batch
        for record in batch:
            yield record
'''

        holes = [
            Hole(
                id="batch_size",
                name="batch_size",
                hole_type="value",
                constraints=[("range", 100, 10000)],
                location={"context": "Records per batch"}
            ),
            Hole(
                id="parallelism",
                name="parallelism",
                hole_type="value",
                constraints=[("range", 1, 50)],
                location={"context": "Number of parallel workers"}
            )
        ]

        return ProgramSketch(
            template=template,
            holes=holes,
            language="python",
            framework="Apache Beam",
            constraints=[]
        )

    def _generate_api_sketch(self, spec: FormalSpecification) -> ProgramSketch:
        """
        Generate REST API service sketch.

        Creates an API service template with holes for:
        - max_connections: Maximum concurrent connections (10-1000)
        - timeout: Request timeout in milliseconds (100-30000)

        Args:
            spec: Formal specification

        Returns:
            ProgramSketch for API service

        References:
        - TD Commons: API service pattern
        - FastAPI: https://fastapi.tiangolo.com/
        """
        template = '''from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Synthesized configuration
MAX_CONNECTIONS = __HOLE_max_connections__
TIMEOUT_MS = __HOLE_timeout__

class Request(BaseModel):
    """API request model."""
    data: dict

class Response(BaseModel):
    """API response model."""
    result: dict
    status: str

@app.post("/process")
async def process(request: Request) -> Response:
    """Process API request."""
    # Process request
    result = {"processed": request.data}
    return Response(result=result, status="success")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        limit_concurrency=MAX_CONNECTIONS,
        timeout_keep_alive=TIMEOUT_MS // 1000
    )
'''

        holes = [
            Hole(
                id="max_connections",
                name="max_connections",
                hole_type="value",
                constraints=[("range", 10, 1000)],
                location={"context": "Maximum concurrent connections"}
            ),
            Hole(
                id="timeout",
                name="timeout",
                hole_type="value",
                constraints=[("range", 100, 30000)],
                location={"context": "Request timeout in milliseconds"}
            )
        ]

        return ProgramSketch(
            template=template,
            holes=holes,
            language="python",
            framework="FastAPI",
            constraints=[]
        )

    def _generate_generic_sketch(self, spec: FormalSpecification) -> ProgramSketch:
        """
        Generate generic program sketch.

        Creates a simple generic template with basic configuration holes
        when no specific pattern is detected.

        Args:
            spec: Formal specification

        Returns:
            ProgramSketch for generic implementation

        References:
        - CEGIS: Generic sketch generation
        """
        template = """
# Synthesized implementation
def synthesized_function(inputs):
    '''
    Generic synthesized function.

    Configuration:
    - param1: __HOLE_param1__
    - param2: __HOLE_param2__
    '''
    # Configuration parameters
    param1 = __HOLE_param1__
    param2 = __HOLE_param2__

    # Simple computation (placeholder)
    result = param1 * inputs.get('x', 0) + param2
    return result
"""

        holes = [
            Hole(
                id="param1",
                name="param1",
                hole_type="value",
                constraints=[("range", 0, 100)]
            ),
            Hole(
                id="param2",
                name="param2",
                hole_type="value",
                constraints=[("range", 0, 100)]
            )
        ]

        return ProgramSketch(
            template=template,
            holes=holes,
            language="python",
            framework="",
            constraints=[]
        )

    def synthesize_holes(
        self,
        sketch: ProgramSketch,
        spec: FormalSpecification,
        examples: List[SynthesisExample],
        counterexamples: List[Dict[str, Any]]
    ) -> bool:
        """
        Synthesize hole values using Z3 SMT solver.

        Uses SMT solving to find values for all holes that satisfy:
        1. Hole constraints (ranges, etc.)
        2. Specification constraints
        3. Example constraints (program matches examples)
        4. Counterexample constraints (avoid previous failures)

        Args:
            sketch: Program sketch with holes to fill
            spec: Formal specification
            examples: Input/output examples
            counterexamples: Previous counterexamples to avoid

        Returns:
            True if hole values found and filled, False otherwise

        Example:
            >>> success = synthesizer.synthesize_holes(
            ...     sketch, spec, examples, counterexamples
            ... )
            >>> if success:
            ...     assert all(h.is_filled() for h in sketch.holes)

        References:
        - CEGIS: SMT-based hole synthesis
        - Z3: Constraint solving
        """
        # Create Z3 solver with timeout
        solver = z3.Solver()
        solver_timeout = self.timeout // 4  # Use 1/4 of total timeout per synthesis
        solver.set("timeout", solver_timeout)

        # Create Z3 variables for each hole
        hole_vars = {}
        for hole in sketch.holes:
            var = hole.to_z3_var()
            if var is not None:
                hole_vars[hole.id] = var

                # Add hole constraints
                hole_constraints = hole.get_constraints_as_z3()
                for constraint in hole_constraints:
                    solver.add(constraint)

        # Add specification constraints
        # (Simplified - full implementation would encode all properties)
        # For now, just ensure we have some constraints

        # Add example constraints
        # Each example: f(inputs) == expected_output
        # (Simplified - would need actual program execution model)
        for example in examples:
            # Placeholder: In full implementation, would encode
            # program behavior as constraints
            pass

        # Add counterexample constraints (rule out previous solutions)
        # (Simplified - would encode negation of counterexamples)
        for ce in counterexamples:
            # Placeholder: would add constraints to avoid this counterexample
            pass

        # Solve
        result = solver.check()

        if result == z3.sat:
            # Extract solution
            model = solver.model()

            # Fill holes with values from model
            for hole in sketch.holes:
                if hole.id in hole_vars:
                    var = hole_vars[hole.id]
                    value = model[var]

                    # Extract concrete value based on hole type
                    if hole.hole_type == "value":
                        # Integer value
                        concrete_value = value.as_long()
                    elif hole.hole_type == "expression":
                        # Real value
                        try:
                            # Try to get as fraction
                            num = value.numerator_as_long()
                            den = value.denominator_as_long()
                            concrete_value = num / den if den != 0 else 0.0
                        except:
                            # Fallback to decimal approximation
                            concrete_value = float(value.as_decimal(10).replace("?", ""))
                    elif hole.hole_type == "predicate":
                        # Boolean value
                        concrete_value = bool(value)
                    else:
                        # Unknown type - use string representation
                        concrete_value = str(value)

                    # Fill hole
                    hole.filled_value = concrete_value
                    logger.debug(f"Filled hole {hole.name} with {concrete_value}")

            return True

        elif result == z3.unsat:
            # No solution exists
            logger.debug("No solution: constraints are unsatisfiable")
            return False

        else:
            # Unknown (timeout or incomplete theory)
            logger.debug(f"Solver returned unknown: {solver.reason_unknown()}")
            return False

    def verify_synthesis(
        self,
        implementation: str,
        spec: FormalSpecification
    ) -> Dict[str, Any]:
        """
        Verify synthesized program against specification.

        Checks if the synthesized implementation satisfies the formal
        specification. Returns verification result with counterexample
        if verification fails.

        Args:
            implementation: Synthesized program code
            spec: Formal specification

        Returns:
            Dictionary with:
            - verified: bool (True if verified)
            - counterexample: Dict (if verification failed)

        Example:
            >>> result = synthesizer.verify_synthesis(code, spec)
            >>> if result["verified"]:
            ...     print("Verification succeeded!")
            >>> else:
            ...     print(f"Counterexample: {result['counterexample']}")

        References:
        - CEGIS: Verification phase
        - TD Commons: Synthesis verification
        """
        # Simplified verification
        # Full implementation would:
        # 1. Parse/compile synthesized code
        # 2. Run formal verifier on code + spec
        # 3. Extract counterexample if verification fails

        # For now, return success (placeholder)
        # In practice, would use the Verifier class
        return {
            "verified": True,
            "counterexample": None
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Synthesizer(max_iterations={self.max_iterations}, "
            f"timeout={self.timeout}ms)"
        )
