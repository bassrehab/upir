"""
UPIR Patterns Module.

Pattern extraction and management for reusable architectural templates.

Author: Subhadip Mitra
License: Apache 2.0
"""

from upir.patterns.extractor import PatternExtractor
from upir.patterns.library import PatternLibrary
from upir.patterns.pattern import Pattern

__all__ = ["Pattern", "PatternExtractor", "PatternLibrary"]
