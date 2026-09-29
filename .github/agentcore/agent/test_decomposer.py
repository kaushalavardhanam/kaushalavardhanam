"""Unit tests for the decomposer's robust JSON extraction.

Runs fully offline (no Bedrock, no AWS). Focuses on the model-response JSON
parsing that broke on issue #13: Sonnet returned a ``files{}`` object whose file
*content* contained raw newlines / control characters, and the old strict
``json.loads`` rejected it with "Invalid control character" / "Unterminated
string". These tests pin the tolerant behaviour and guard the well-formed path.

    python -m unittest test_decomposer
"""

from __future__ import annotations

import json
import unittest

import decomposer
from decomposer import (
    _escape_control_chars_in_strings,
    _extract_json_array,
    _extract_json_object,
    _loads_tolerant,
)


class TestRobustJsonExtraction(unittest.TestCase):
    # ---- the exact issue #13 failure mode -------------------------------- #

    def test_object_with_raw_newlines_in_string_value(self):
        """A files{} payload whose content has REAL newlines must parse.

        This is what Sonnet returned for issue #13 (a SETUP_LOG.md with raw
        line breaks). Building the JSON by hand with literal '\n' inside the
        string value reproduces the control character strict JSON rejects.
        """
        content = "# MITRA Coherence Check\nline two\nline three"
        # Hand-built JSON with the newline left LITERAL inside the string value.
        raw = '{"files": {"docs/SETUP_LOG.md": "' + content + '"}, "note": "wrote log"}'
        # Sanity: this is exactly what strict json.loads chokes on.
        with self.assertRaises(json.JSONDecodeError):
            json.loads(raw)
        # The tolerant extractor must recover it and preserve the newlines.
        result = _extract_json_object(raw)
        self.assertEqual(result["files"]["docs/SETUP_LOG.md"], content)
        self.assertEqual(result["note"], "wrote log")

    def test_object_with_tab_and_other_control_char(self):
        """Literal tab and a bare control char (e.g. 0x0b) inside a value."""
        content = "col1\tcol2\x0bmore"
        raw = '{"files": {"a.txt": "' + content + '"}}'
        with self.assertRaises(json.JSONDecodeError):
            json.loads(raw)
        result = _extract_json_object(raw)
        self.assertEqual(result["files"]["a.txt"], content)

    def test_escape_helper_only_touches_inside_strings(self):
        """Structural newlines between tokens are untouched; in-string ones escaped."""
        raw = '{\n  "k": "a\nb"\n}'  # newline after { is structural; the one in "a\nb" is not
        repaired = _escape_control_chars_in_strings(raw)
        parsed = json.loads(repaired)  # strict parse now succeeds
        self.assertEqual(parsed["k"], "a\nb")

    def test_escape_helper_respects_backslash_escapes(self):
        """An already-escaped \\n must not be doubled or misread as string end."""
        raw = '{"k": "already\\nescaped"}'
        # Well-formed already; tolerant loader returns it unchanged in meaning.
        self.assertEqual(_loads_tolerant(raw)["k"], "already\nescaped")

    # ---- well-formed responses: behaviour unchanged ---------------------- #

    def test_wellformed_object_unchanged(self):
        raw = '{"files": {"x.py": "print(1)\\n"}, "note": "ok"}'
        self.assertEqual(_extract_json_object(raw), json.loads(raw))

    def test_wellformed_array_unchanged(self):
        raw = '[{"id": 1, "title": "t", "description": "d"}]'
        self.assertEqual(_extract_json_array(raw), json.loads(raw))

    # ---- prose / markdown fence tolerance -------------------------------- #

    def test_object_wrapped_in_json_fence(self):
        raw = 'Here you go:\n```json\n{"files": {"a": "b\nc"}}\n```\nDone.'
        result = _extract_json_object(raw)
        self.assertEqual(result["files"]["a"], "b\nc")

    def test_array_with_leading_prose(self):
        raw = 'Sure:\n[{"id": 1, "title": "x", "description": "y"}]'
        result = _extract_json_array(raw)
        self.assertEqual(result[0]["id"], 1)

    # ---- genuinely unparseable still raises ------------------------------ #

    def test_no_object_raises_valueerror(self):
        with self.assertRaises(ValueError):
            _extract_json_object("no json here at all")

    def test_no_array_raises_valueerror(self):
        with self.assertRaises(ValueError):
            _extract_json_array("no array here")


if __name__ == "__main__":
    unittest.main()
