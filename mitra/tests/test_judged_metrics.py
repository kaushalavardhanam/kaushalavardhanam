"""Tests for eval.judged_metrics using a fake judge (no network)."""

import json
import tempfile
import unittest
from pathlib import Path

from eval import judged_metrics as jm


def _fake_judge_factory(responses):
    calls = []

    def judge(prompt):
        calls.append(prompt)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    judge.calls = calls
    return judge


REPEATS_OK = json.dumps(
    {
        "repeat_count": 1,
        "parroting_count": 0,
        "examples": [{"type": "repeat", "span": "नमस्ते", "earlier_turn": 0, "note": "same"}],
    },
    ensure_ascii=False,
)
HINDI_OK = json.dumps({"hindi_form_count": 0, "examples": []})


class ParsingTests(unittest.TestCase):
    def test_extract_json_with_fences(self):
        obj = jm.extract_json("```json\n{\"a\": 1}\n```")
        self.assertEqual(obj, {"a": 1})

    def test_extract_json_missing(self):
        with self.assertRaises(jm.JudgeFormatError):
            jm.extract_json("no json here")

    def test_bad_count_rejected(self):
        with self.assertRaises(jm.JudgeFormatError):
            jm.validate_hindi({"hindi_form_count": "2", "examples": []})

    def test_count_mismatch_warns(self):
        obj = jm.validate_hindi({"hindi_form_count": 2, "examples": []})
        self.assertIn("warnings", obj)


class JudgeTests(unittest.TestCase):
    def test_prompt_contains_history(self):
        judge = _fake_judge_factory([REPEATS_OK])
        res = jm.judge_repeats(["प्रथमम्", "द्वितीयम्"], "तृतीयम्", judge)
        self.assertEqual(res["repeat_count"], 1)
        self.assertIn("[turn 0] प्रथमम्", judge.calls[0])
        self.assertIn("[turn 1] द्वितीयम्", judge.calls[0])
        self.assertIn("तृतीयम्", judge.calls[0])

    def test_retry_on_malformed(self):
        judge = _fake_judge_factory(["oops", HINDI_OK])
        res = jm.judge_hindi_forms([], "text", judge)
        self.assertEqual(res["hindi_form_count"], 0)
        self.assertEqual(len(judge.calls), 2)

    def test_fails_after_retry(self):
        judge = _fake_judge_factory(["oops"])
        with self.assertRaises(jm.JudgeFormatError):
            jm.judge_repeats([], "text", judge)

    def test_custom_metric(self):
        judge = _fake_judge_factory([json.dumps({"count": 2, "score": None, "examples": []})])
        res = jm.judge_custom("register", "count switches", [], "text", judge)
        self.assertEqual(res["count"], 2)


class RunTests(unittest.TestCase):
    def _records(self):
        return [
            {"category": "c", "seed_question": "q", "run": 1, "depth": 1, "text": "b"},
            {"category": "c", "seed_question": "q", "run": 1, "depth": 0, "text": "a"},
        ]

    def test_history_is_previous_turns_only(self):
        judge = _fake_judge_factory([REPEATS_OK])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "judged.jsonl"
            written = jm.run_judging(self._records(), {"repeats"}, judge, out_path=out)
            self.assertEqual(len(written), 2)
            by_depth = {w["depth"]: w for w in written}
            self.assertEqual(by_depth[0]["history"], [])
            self.assertEqual(by_depth[1]["history"], ["a"])

    def test_skip_already_judged_and_record_errors(self):
        good = _fake_judge_factory([REPEATS_OK])
        bad = _fake_judge_factory(["oops"])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "judged.jsonl"
            jm.run_judging(self._records(), {"repeats"}, good, out_path=out)
            again = jm.run_judging(self._records(), {"repeats"}, good, out_path=out)
            self.assertEqual(again, [])
            out2 = Path(d) / "judged2.jsonl"
            written = jm.run_judging(self._records(), {"repeats"}, bad, out_path=out2)
            self.assertTrue(all(w.get("error") for w in written))

    def test_spotcheck_and_report(self):
        judge = _fake_judge_factory([REPEATS_OK])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "judged.jsonl"
            spot = Path(d) / "spot.jsonl"
            jm.run_judging(self._records(), {"repeats"}, judge, out_path=out)
            rows = jm.draw_spotcheck(jm.load_judged(out), n=2, out_path=spot)
            self.assertEqual(len(rows), 2)
            self.assertIsNone(jm.spotcheck_report(spot)["agreement_rate"])
            saved = [json.loads(l) for l in spot.read_text(encoding="utf-8").splitlines()]
            saved[0]["manual_agree"] = True
            saved[1]["manual_agree"] = False
            spot.write_text("\n".join(json.dumps(s) for s in saved), encoding="utf-8")
            self.assertEqual(jm.spotcheck_report(spot)["agreement_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()