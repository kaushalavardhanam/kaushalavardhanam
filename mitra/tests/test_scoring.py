import unittest

from eval.scoring import (
    ComponentSpec,
    build_scheme,
    mean_score,
    normalize_score,
)


def _specs():
    specs = []
    for i in range(10):
        specs.append(
            ComponentSpec(
                f"c{i}",
                "-" if i % 2 else "+",
                in_purity=i < 7,
            )
        )
    return specs


class NormalizeTests(unittest.TestCase):
    def test_higher_is_better(self):
        spec = ComponentSpec("a", "+", 0, 10)
        self.assertAlmostEqual(normalize_score(2.5, spec), 0.25)

    def test_lower_is_better_flips(self):
        spec = ComponentSpec("a", "-", 0, 1)
        self.assertAlmostEqual(normalize_score(0.2, spec), 0.8)

    def test_clamps(self):
        spec = ComponentSpec("a", "+", 0, 1)
        self.assertEqual(normalize_score(5, spec), 1.0)
        self.assertEqual(normalize_score(-5, spec), 0.0)

    def test_missing_raises(self):
        with self.assertRaises(ValueError):
            normalize_score(None, ComponentSpec("a"))

    def test_bad_direction(self):
        with self.assertRaises(ValueError):
            ComponentSpec("a", "x")

    def test_mean_score(self):
        self.assertAlmostEqual(mean_score([1, None, 0]), 0.5)
        self.assertIsNone(mean_score([None]))


class SchemeTests(unittest.TestCase):
    def test_counts_enforced(self):
        with self.assertRaises(ValueError):
            build_scheme(_specs()[:9])
        bad = [ComponentSpec(f"c{i}", in_purity=i < 6) for i in range(10)]
        with self.assertRaises(ValueError):
            build_scheme(bad)

    def test_performance(self):
        scheme = build_scheme(_specs())
        # "+" components get 1.0, "-" components get 0.0 -> all normalize to 1.0
        raw = {f"c{i}": (0.0 if i % 2 else 1.0) for i in range(10)}
        self.assertAlmostEqual(scheme.coherence_performance(raw), 1.0)
        self.assertAlmostEqual(scheme.purity_performance(raw), 1.0)

    def test_purity_uses_only_purity_components(self):
        scheme = build_scheme(_specs())
        raw = {f"c{i}": 1.0 for i in range(10)}
        # purity: c0..c6, "-" at 1,3,5 -> normalized 0; "+" at 0,2,4,6 -> 1
        self.assertAlmostEqual(scheme.purity_performance(raw), 4 / 7)
        # coherence: "+" at 0,2,4,6,8 -> 1 (5 of 10)
        self.assertAlmostEqual(scheme.coherence_performance(raw), 0.5)

    def test_missing_component(self):
        scheme = build_scheme(_specs())
        with self.assertRaises(ValueError):
            scheme.coherence_performance({"c0": 1.0})


if __name__ == "__main__":
    unittest.main()