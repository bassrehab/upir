"""
UPIR: Universal Plan Intermediate Representation

Formal verification, synthesis, and optimization for distributed systems.

Based on TD Commons disclosure:
https://www.tdcommons.org/dpubs_series/8852/

Author: Subhadip Mitra
License: Apache 2.0
"""

__version__ = "0.1.0"
__author__ = "Subhadip Mitra"
__license__ = "Apache 2.0"

# Main exports
from upir.core.architecture import Architecture
from upir.core.specification import FormalSpecification
from upir.core.temporal import TemporalOperator, TemporalProperty
from upir.core.upir import UPIR
from upir.learning.learner import ArchitectureLearner
from upir.patterns.extractor import PatternExtractor
from upir.patterns.library import PatternLibrary
from upir.patterns.pattern import Pattern
from upir.synthesis.cegis import Synthesizer
from upir.verification.verifier import Verifier

__all__ = [
    # Core classes
    "UPIR",
    "Architecture",
    "FormalSpecification",
    "TemporalOperator",
    "TemporalProperty",
    # Verification
    "Verifier",
    # Synthesis
    "Synthesizer",
    # Learning
    "ArchitectureLearner",
    # Patterns
    "Pattern",
    "PatternExtractor",
    "PatternLibrary",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]
