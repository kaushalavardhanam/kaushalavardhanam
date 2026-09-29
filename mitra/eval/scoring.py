"""Score normalization and performance formulas for the MITRA coherence check.

Terminology
-----------
score        a single component's value (e.g. one component measured on one
             reply, or averaged over runs). Components are first reduced to a
             *normalized score* in [0, 1] where HIGHER IS ALWAYS BETTER.
performance  the combined result of a test: the equal-weight mean of the
             normalized scores of several components.

    coherence performance = mean(normalized score of all 10 components)
    purity performance    = mean(normalized score of the 7 purity components)

Normalization (per component)
-----------------------------
Each component is declared with a ComponentSpec giving the raw range
[min_value, max_value] and a direction:

    "+"  higher raw value is better:  norm = (raw - min) / (max - min)
    "-"  lower raw value is better:   norm = 1 - (raw - min) / (max - min)

Raw values outside the range are clamped. A boolean/rate component uses the
default range 0..1 (a rate of 0.2 with direction "-" gives 0.8).

The component names are NOT hard-coded here: build the scheme from the
component list in the issue with build_scheme(), which checks that there are
exactly 10 components and 7 purity components.

Usage (from the mitra/ directory):

    from eval.scoring import ComponentSpec, build_scheme

    scheme = build_scheme([
        ComponentSpec("some_component", "+", in_purity=True),
        ComponentSpec("some_penalty", "-"),
        ...  # 10 in total, 7 with in_purity=True
    ])
    scheme.coherence_performance(raw_scores)   # float in [0, 1]
    scheme.purity_performance(raw_scores)      # float in [0, 1]
"""

import math
from dataclasses import dataclass

HIGHER_IS_BETTER = "+"
LOWER_IS_BETTER = "-"

EXPECTED_COMPONENTS = 10
EXPECTED_PURITY_COMPONENTS = 7


@dataclass(frozen=True)
class ComponentSpec:
    """Declaration of one component: name, direction, raw range, purity membership."""

    name: str
    direction: str = HIGHER_IS_BETTER
    min_value: float = 0.0
    max_value: float = 1.0
    in_purity: bool = False

    def __post_init__(self):
        if self.direction not in (HIGHER_IS_BETTER, LOWER_IS_BETTER):
            raise ValueError(
                f"{self.name}: direction must be '+' or '-', got {self.direction!r}"
            )
        if not self.max_value > self.min_value:
            raise ValueError(f"{self.name}: max_value must be greater than min_value")


def normalize_score(raw, spec):
    """Map a raw component value to [0, 1], higher always better."""
    if raw is None:
        raise ValueError(f"{spec.name}: raw score is missing")
    value = float(raw)
    if math.isnan(value):
        raise ValueError(f"{spec.name}: raw score is NaN")
    value = min(max(value, spec.min_value), spec.max_value)
    fraction = (value - spec.min_value) / (spec.max_value - spec.min_value)
    return fraction if spec.direction == HIGHER_IS_BETTER else 1.0 - fraction


def mean_score(values):
    """Average raw values for one component across runs, ignoring None."""
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def combine(normalized):
    """Equal-weight combination (arithmetic mean) of normalized scores."""
    normalized = list(normalized)
    if not normalized:
        raise ValueError("no scores to combine")
    return sum(normalized) / len(normalized)


class ScoringScheme:
    """A validated set of component specs with the two performance formulas."""

    def __init__(self, specs, expected_total=EXPECTED_COMPONENTS,
                 expected_purity=EXPECTED_PURITY_COMPONENTS):
        specs = tuple(specs)
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError("component names must be unique")
        purity_count = sum(1 for s in specs if s.in_purity)
        if expected_total is not None and len(specs) != expected_total:
            raise ValueError(
                f"expected {expected_total} components, got {len(specs)}"
            )
        if expected_purity is not None and purity_count != expected_purity:
            raise ValueError(
                f"expected {expected_purity} purity components, got {purity_count}"
            )
        self.specs = specs

    @property
    def purity_specs(self):
        return tuple(s for s in self.specs if s.in_purity)

    def normalized_scores(self, raw_scores, specs=None):
        """Return {name: normalized score} for the given specs (default: all)."""
        result = {}
        for spec in specs if specs is not None else self.specs:
            if spec.name not in raw_scores:
                raise ValueError(f"missing raw score for component {spec.name!r}")
            result[spec.name] = normalize_score(raw_scores[spec.name], spec)
        return result

    def coherence_performance(self, raw_scores):
        """Equal-weight mean of all components' normalized scores."""
        return combine(self.normalized_scores(raw_scores).values())

    def purity_performance(self, raw_scores):
        """Equal-weight mean of the purity components' normalized scores."""
        return combine(
            self.normalized_scores(raw_scores, self.purity_specs).values()
        )


def build_scheme(specs, expected_total=EXPECTED_COMPONENTS,
                 expected_purity=EXPECTED_PURITY_COMPONENTS):
    """Create a ScoringScheme, enforcing 10 components / 7 purity by default."""
    return ScoringScheme(specs, expected_total, expected_purity)