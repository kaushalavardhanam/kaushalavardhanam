"""Unit tests for the Claude session guard and Bedrock env (``claude_runner.py``).

Offline: exercises the pure policy functions the PreToolUse hook delegates to.

    python -m unittest test_claude_runner
"""

from __future__ import annotations

import os
import tempfile
import unittest

from claude_runner import ClaudeRunner, bedrock_env, check_bash, check_tool


class BashPolicyTests(unittest.TestCase):
    def test_allows_reading_and_testing(self):
        for cmd in ("ls -la mitra/tests", "grep -rn Vocabulary mitra/src | head -20",
                    "cd mitra && python -m pytest tests/test_vocabulary.py -q 2>&1 | tail -30",
                    "git diff HEAD", "git log --oneline -5", "PYTHONPATH=. pytest -q",
                    "find . -name '*.py' -path '*lexicon*'"):
            with self.subTest(cmd=cmd):
                self.assertIsNone(check_bash(cmd, allow_tests=True))

    def test_denies_writes_network_git_writes_and_installs(self):
        for cmd in ("git commit -am x", "git push origin HEAD", "curl https://x", "rm -rf /",
                    "echo hi > file.txt", "cat a >> b", "pip install requests",
                    "python -m pip install x", "ls $(whoami)", "ls `id`",
                    "find . -name x -delete", "find . -exec rm {} ;", "wget x"):
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(check_bash(cmd, allow_tests=True))

    def test_tests_can_be_disallowed(self):
        self.assertIsNotNone(check_bash("pytest -q", allow_tests=False))
        self.assertIsNone(check_bash("ls", allow_tests=False))


class ToolPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cwd = tempfile.mkdtemp()

    def test_read_only_session_cannot_write(self):
        reason = check_tool("Edit", {"file_path": os.path.join(self.cwd, "a.py")},
                            cwd=self.cwd, write_paths=None, allow_tests=True)
        self.assertIn("read-only", reason)

    def test_writes_limited_to_declared_files(self):
        allowed = ["mitra/tests/test_vocabulary.py"]
        ok = check_tool("Write", {"file_path": os.path.join(self.cwd, allowed[0])},
                        cwd=self.cwd, write_paths=allowed, allow_tests=True)
        self.assertIsNone(ok)
        for target in ("mitra/src/lexicon/vocabulary.py", "../escape.py",
                       os.path.join(self.cwd, "mitra/tests/../src/x.py")):
            with self.subTest(target=target):
                self.assertIsNotNone(check_tool("Edit", {"file_path": target}, cwd=self.cwd,
                                                write_paths=allowed, allow_tests=True))

    def test_web_and_subagent_tools_blocked(self):
        for tool in ("WebFetch", "WebSearch", "Task", "Agent"):
            self.assertIsNotNone(check_tool(tool, {}, cwd=self.cwd, write_paths=[], allow_tests=True))


class BedrockEnvTests(unittest.TestCase):
    def test_defaults_to_claude_on_bedrock(self):
        env = bedrock_env("global.anthropic.claude-sonnet-5-5")
        self.assertEqual(env["CLAUDE_CODE_USE_BEDROCK"], "1")
        self.assertEqual(env["ANTHROPIC_MODEL"], env["ANTHROPIC_SMALL_FAST_MODEL"])

    def test_rejects_non_anthropic_models(self):
        with self.assertRaises(ValueError):
            bedrock_env("global.openai.gpt-6-astra")


class RunnerEnvTests(unittest.TestCase):
    MODEL = "global.anthropic.claude-sonnet-5-5"

    def test_scrubs_subprocess_env(self):
        runner = ClaudeRunner(tempfile.mkdtemp(), model_id=self.MODEL)
        self.assertEqual(runner.env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"], "1")

    def test_extra_env_cannot_disable_scrub(self):
        runner = ClaudeRunner(tempfile.mkdtemp(), model_id=self.MODEL,
                              extra_env={"CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "0", "FOO": "bar"})
        self.assertEqual(runner.env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"], "1")
        self.assertEqual(runner.env["FOO"], "bar")


if __name__ == "__main__":
    unittest.main()
