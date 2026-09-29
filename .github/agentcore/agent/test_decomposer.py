"""Unit tests for the decomposer's sentinel-delimited implement contract.

Runs fully offline (no Bedrock, no AWS). These pin the DURABLE fix for issue
#13: file contents are returned as RAW text between ``<<<FILE ...>>>`` /
``<<<END FILE>>>`` sentinels rather than as JSON string values, so an unescaped
quote / newline / control char inside a file body can no longer break parsing,
and a body truncated at the token cap (missing its ``<<<END FILE>>>``) is
detected and continued per-file instead of failing the whole sub-task.

    python -m unittest test_decomposer
"""

from __future__ import annotations

import json
import unittest

import decomposer
from decomposer import (
    MAX_FILE_CONTINUATIONS,
    SubTask,
    _extract_json_array,
    _parse_sentinel_files,
    _sanitize_path,
    run_decomposition,
)


# --------------------------------------------------------------------------- #
# Fake Bedrock client
# --------------------------------------------------------------------------- #


class FakeBedrock:
    """Scripted stand-in for BedrockClient.

    ``decompose_response`` is returned (via ``invoke``) for the decompose call.
    ``implement_responses`` is a list consumed in order by ``invoke_partial``;
    each entry is ``(text, truncated)``. This lets a test script a truncated
    first implement response followed by a completing continuation.
    """

    def __init__(self, decompose_response, implement_responses):
        self._decompose_response = decompose_response
        self._implement = list(implement_responses)
        self.invoke_calls = []
        self.invoke_partial_calls = []

    def invoke(self, prompt, *, system=None, temperature=0.2, max_tokens=None):
        self.invoke_calls.append(prompt)
        return self._decompose_response

    def invoke_partial(self, prompt, *, system=None, temperature=0.2, max_tokens=None):
        self.invoke_partial_calls.append(prompt)
        if not self._implement:
            raise AssertionError("invoke_partial called more times than scripted")
        return self._implement.pop(0)


def _one_subtask_decompose():
    return json.dumps([{"id": 1, "title": "Do the thing", "description": "make it so"}])


# --------------------------------------------------------------------------- #
# Sentinel parser — the core of the durable fix
# --------------------------------------------------------------------------- #


class TestSentinelParser(unittest.TestCase):

    def test_issue13_unescaped_quote_in_large_body_parses_clean(self):
        """The exact #13 sub-task-5 failure mode: a file body containing an
        unescaped double-quote (and raw newlines) — structurally impossible to
        embed in JSON reliably — is now taken VERBATIM and parses cleanly."""
        body = (
            '#!/usr/bin/env python3\n'
            '"""Env check for MITRA."""\n'
            'MSG = "he said \\"hello\\" and left"\n'
            'RAW = "an unescaped \" right here breaks JSON"\n'
            "print('done')\n"
            + "# padding line\n" * 400  # make it a LARGE body like #13
        )
        raw = f'<<<FILE path="mitra/tools/check_env.py">>>\n{body}\n<<<END FILE>>>\n'
        files, note = _parse_sentinel_files(raw)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].path, "mitra/tools/check_env.py")
        self.assertFalse(files[0].truncated)
        # Byte-for-byte: the raw quote survives, nothing was unescaped.
        self.assertEqual(files[0].body, body)
        self.assertIn('unescaped " right here', files[0].body)

    def test_truncated_body_missing_end_sentinel_is_detected(self):
        """A file whose <<<END FILE>>> is missing (cut at the token cap) is
        flagged truncated rather than silently accepted."""
        raw = (
            '<<<FILE path="a/big.py">>>\n'
            'def f():\n    return "partial content that got cut'
        )  # no closing sentinel
        files, note = _parse_sentinel_files(raw)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].truncated)
        self.assertIn("partial content", files[0].body)

    def test_multi_file_roundtrip_byte_for_byte(self):
        """A normal multi-file response round-trips each body exactly."""
        b1 = "line1\nline2\n\ttabbed\n"
        b2 = 'print("hi")\n# trailing comment'
        raw = (
            f'<<<FILE path="src/one.txt">>>\n{b1}\n<<<END FILE>>>\n'
            f'<<<FILE path="src/two.py">>>\n{b2}\n<<<END FILE>>>\n'
            "<<<NOTE>>>added two files<<<END NOTE>>>"
        )
        files, note = _parse_sentinel_files(raw)
        self.assertEqual([f.path for f in files], ["src/one.txt", "src/two.py"])
        self.assertEqual(files[0].body, b1)
        self.assertEqual(files[1].body, b2)
        self.assertFalse(files[0].truncated)
        self.assertFalse(files[1].truncated)
        self.assertEqual(note, "added two files")

    def test_body_containing_json_and_code_fences_is_verbatim(self):
        """A body that itself contains JSON braces and ``` fences (which broke
        the old fence-stripping JSON parser) is untouched."""
        body = (
            'Here is JSON: {"files": {"x": "y"}}\n'
            "```bash\necho hello\n```\n"
            "and a lone <<<FILE marker-ish text but not a real header\n"
        )
        raw = f'<<<FILE path="docs/readme.md">>>\n{body}\n<<<END FILE>>>\n'
        files, _ = _parse_sentinel_files(raw)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].body, body)

    def test_bare_path_header_form_accepted(self):
        raw = '<<<FILE scripts/run.sh>>>\n#!/bin/sh\necho hi\n<<<END FILE>>>\n'
        files, _ = _parse_sentinel_files(raw)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].path, "scripts/run.sh")
        self.assertEqual(files[0].body, "#!/bin/sh\necho hi")


# --------------------------------------------------------------------------- #
# End-to-end workflow: truncation -> per-file continuation -> stitch
# --------------------------------------------------------------------------- #


class TestImplementContinuation(unittest.TestCase):

    def test_truncated_file_is_continued_and_stitched(self):
        """A first implement response truncated mid-file triggers a per-file
        re-ask; the continuation body is stitched onto the partial."""
        # First implement call: file opened, body cut off, NO end sentinel.
        part1 = '<<<FILE path="big.py">>>\ndef f():\n    x = "start of a big file'
        # Continuation call: the remaining content, properly terminated.
        part2 = '<<<FILE path="big.py">>>\n end of the big file"\n    return x\n<<<END FILE>>>\n'
        fake = FakeBedrock(
            decompose_response=_one_subtask_decompose(),
            implement_responses=[(part1, True), (part2, False)],
        )
        result = run_decomposition(
            issue_number=13, title="t", body="b", repo="o/r",
            base_branch="main", bedrock=fake,
        )
        self.assertIn("big.py", result["files"])
        stitched = result["files"]["big.py"]
        # The stitched file contains both halves, verbatim, joined at the cut.
        self.assertEqual(
            stitched,
            'def f():\n    x = "start of a big file end of the big file"\n    return x',
        )
        # Exactly one initial implement + one continuation call.
        self.assertEqual(len(fake.invoke_partial_calls), 2)

    def test_untruncated_multi_file_needs_no_continuation(self):
        resp = (
            '<<<FILE path="a.py">>>\nprint(1)\n<<<END FILE>>>\n'
            '<<<FILE path="b.py">>>\nprint(2)\n<<<END FILE>>>\n'
            "<<<NOTE>>>two files<<<END NOTE>>>"
        )
        fake = FakeBedrock(
            decompose_response=_one_subtask_decompose(),
            implement_responses=[(resp, False)],
        )
        result = run_decomposition(
            issue_number=1, title="t", body="b", repo="o/r",
            base_branch="main", bedrock=fake,
        )
        self.assertEqual(result["files"], {"a.py": "print(1)", "b.py": "print(2)"})
        # No continuation call was made.
        self.assertEqual(len(fake.invoke_partial_calls), 1)

    def test_no_file_blocks_is_a_step_failure(self):
        """If the model ignores the sentinel contract entirely, the step errors
        (rather than silently producing nothing)."""
        fake = FakeBedrock(
            decompose_response=_one_subtask_decompose(),
            implement_responses=[("I could not do this task, sorry.", False)],
        )
        with self.assertRaises(RuntimeError) as ctx:
            run_decomposition(
                issue_number=1, title="t", body="b", repo="o/r",
                base_branch="main", bedrock=fake,
            )
        self.assertIn("no <<<FILE>>> blocks", str(ctx.exception))

    def test_continuation_is_bounded(self):
        """A file that stays truncated forever stops after the bound and returns
        best-effort stitched content instead of looping."""
        # 1 initial + MAX_FILE_CONTINUATIONS continuations, all truncated.
        responses = [('<<<FILE path="x">>>\nchunk0', True)]
        responses += [
            (f'<<<FILE path="x">>>\nchunk{i}', True)
            for i in range(1, MAX_FILE_CONTINUATIONS + 1)
        ]
        fake = FakeBedrock(
            decompose_response=_one_subtask_decompose(),
            implement_responses=responses,
        )
        result = run_decomposition(
            issue_number=1, title="t", body="b", repo="o/r",
            base_branch="main", bedrock=fake,
        )
        # It stitched every chunk and stopped (did not exceed the bound).
        self.assertIn("x", result["files"])
        self.assertTrue(result["files"]["x"].startswith("chunk0"))
        self.assertEqual(
            len(fake.invoke_partial_calls), 1 + MAX_FILE_CONTINUATIONS
        )


# --------------------------------------------------------------------------- #
# Decompose still uses JSON (metadata only) — keep that path covered
# --------------------------------------------------------------------------- #


class TestDecomposeJsonArray(unittest.TestCase):

    def test_plain_array(self):
        raw = '[{"id": 1, "title": "a", "description": "d"}]'
        self.assertEqual(_extract_json_array(raw)[0]["title"], "a")

    def test_array_wrapped_in_fence(self):
        raw = '```json\n[{"id": 1, "title": "a", "description": "d"}]\n```'
        self.assertEqual(len(_extract_json_array(raw)), 1)

    def test_array_with_surrounding_prose(self):
        raw = 'Here you go:\n[{"id": 1, "title": "a", "description": "d"}]\nDone.'
        self.assertEqual(_extract_json_array(raw)[0]["id"], 1)

    def test_no_array_raises(self):
        with self.assertRaises(ValueError):
            _extract_json_array("no array here")


class TestSanitizePath(unittest.TestCase):

    def test_rejects_traversal(self):
        self.assertIsNone(_sanitize_path("../etc/passwd"))

    def test_strips_leading_slash(self):
        self.assertEqual(_sanitize_path("/abs/path.py"), "abs/path.py")

    def test_normal_path(self):
        self.assertEqual(_sanitize_path("a/b/c.py"), "a/b/c.py")


if __name__ == "__main__":
    unittest.main()
