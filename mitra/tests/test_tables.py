import unittest

from eval.tables import build_report, collect, format_cell, render_table


RECORDS = [
    {"depth": 0, "component": "greeting", "coherence": 1.0, "purity": 0.5},
    {"depth": 0, "component": "greeting", "coherence": 0.0, "purity": 0.5},
    {"depth": 0, "component": "travel", "coherence": 0.5, "purity": 1.0},
    {"depth": 1, "component": "greeting", "coherence": 0.5, "purity": None},
]


class TablesTest(unittest.TestCase):
    def test_format_cell(self):
        self.assertEqual(format_cell([]), "-")
        self.assertEqual(format_cell([0.5]), "0.500 ± -")
        self.assertEqual(format_cell([0.0, 1.0]), "0.500 ± 0.707")

    def test_collect_skips_missing(self):
        cells = collect(RECORDS, "purity")
        self.assertNotIn((1, "greeting"), cells)
        self.assertEqual(cells[(0, "greeting")], [0.5, 0.5])

    def test_table_has_subtotals(self):
        cells = collect(RECORDS, "coherence")
        table = render_table(cells, [0, 1], ["greeting", "travel"])
        lines = table.splitlines()
        self.assertIn("All components", lines[0])
        self.assertIn("All depths", lines[-1])
        # depth 0 subtotal pools 1.0, 0.0, 0.5 -> mean 0.5
        self.assertIn("0.500 ± 0.500", lines[2])

    def test_report_with_comparison(self):
        report = build_report(RECORDS, comparison=RECORDS)
        self.assertIn("## Coherence - MITRA", report)
        self.assertIn("## Purity - GPT5.6 comparison", report)


if __name__ == "__main__":
    unittest.main()
</gr_replace>