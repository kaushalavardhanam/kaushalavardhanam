"""Sanskrit morphology: attestation, lemmas, and agreement (DESIGN §5).

Wraps the vidyut kosha — a 30M-form inflected lexicon with Pāṇinian
morphology — so the pipeline can ask two questions the Devanagari script
check cannot: "is this a Sanskrit word at all?" and "do these words agree?"
"""

from . import grammar  # noqa: F401
from .analyzer import Analysis, Analyzer  # noqa: F401
