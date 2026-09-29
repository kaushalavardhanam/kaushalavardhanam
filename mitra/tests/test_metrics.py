"""Unit tests for eval.metrics. Run from mitra/: python -m unittest tests.test_metrics"""

import unittest

from eval import metrics


class TestSingleReplyMetrics(unittest.TestCase):
    def test_empty(self):
        self.assertTrue(metrics.is_empty(None))
        self.assertTrue(metrics.is_empty("   \n"))
        self.assertFalse(metrics.is_empty("नमस्ते"))

    def test_word_count_ignores_danda(self):
        self.assertEqual(metrics.word_count("सः गृहं गच्छति ।"), 3)

    def test_latin_detection(self):
        self.assertTrue(metrics.has_latin("अहं hello गच्छामि"))
        self.assertFalse(metrics.has_latin("अहं गच्छामि ।"))

    def test_repeated_4grams(self):
        self.assertEqual(metrics.repeated_4grams("a b c d a b c d"), 1)
        self.assertEqual(metrics.repeated_4grams("अहं गृहं गच्छामि"), 0)
        self.assertEqual(
            metrics.repeated_4grams("क ख ग घ ङ च क ख ग घ"), 1
        )

    def test_pronoun_verb_agreement(self):
        self.assertEqual(metrics.pronoun_verb_errors("सः गृहं गच्छति ।"), 0)
        self.assertEqual(metrics.pronoun_verb_errors("अहं पठामि ।"), 0)
        self.assertEqual(metrics.pronoun_verb_errors("अहं गच्छति ।"), 1)
        self.assertEqual(metrics.pronoun_verb_errors("त्वम् गच्छामि ।"), 1)
        # No verb-like word: not counted.
        self.assertEqual(metrics.pronoun_verb_errors("सः बालकः ।"), 0)

    def test_recognition_counts(self):
        known = {"अहं", "गच्छामि"}
        rec, tot = metrics.recognition_counts("अहं गृहं गच्छामि ।", known.__contains__)
        self.assertEqual((rec, tot), (2, 3))


class TestAggregate(unittest.TestCase):
    def test_compute_metrics(self):
        replies = [
            "अहं गच्छामि ।",
            "सः hello गच्छति ।",
            "",
            "क ख ग घ क ख ग घ",
            "अहं गच्छति ।",
        ]
        m = metrics.compute_metrics(
            replies,
            grammar_scores=[1.0, 0.5, 0.0, 0.5, 0.0],
            hindi_flags=[False, False, False, False, True],
            word_lookup={"अहं", "गच्छामि", "सः", "गच्छति"}.__contains__,
        )
        self.assertEqual(m["n_replies"], 5)
        self.assertEqual(m["empty_replies"], 1)
        self.assertEqual(m["replies_with_latin"], 1)
        self.assertEqual(m["replies_with_hindi"], 1)
        self.assertEqual(m["replies_with_4gram_repeat"], 1)
        self.assertEqual(m["pronoun_verb_errors"], 1)
        self.assertAlmostEqual(m["grammar_score_mean"], 0.5)
        # words per non-empty reply: 2, 3, 8, 3 -> median 3
        self.assertEqual(m["median_words"], 3)
        self.assertIsNotNone(m["recognition_rate"])

    def test_no_lookup_gives_no_recognition_rate(self):
        m = metrics.compute_metrics(["अहं गच्छामि"])
        self.assertIsNone(m["recognition_rate"])

    def test_all_empty(self):
        m = metrics.compute_metrics(["", None])
        self.assertEqual(m["empty_replies"], 2)
        self.assertIsNone(m["median_words"])
        self.assertIsNone(m["grammar_score_mean"])


if __name__ == "__main__":
    unittest.main()