"""Tests for the dispatcher Lambda (``dispatcher.py``). Offline; boto3 is faked.

Terraform packages only ``dispatcher.py`` (``archive_file.source_file``), so
this file never ships.

    python -m unittest test_dispatcher
"""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
from unittest import mock

try:  # boto3 ships in the Lambda runtime; stub it only where it isn't installed
    import boto3  # noqa: F401
except ImportError:
    _botocore = types.ModuleType("botocore")
    _exceptions = types.ModuleType("botocore.exceptions")
    _exceptions.BotoCoreError = type("BotoCoreError", (Exception,), {})
    _exceptions.ClientError = type("ClientError", (Exception,), {})
    sys.modules.update({"boto3": types.ModuleType("boto3"), "botocore": _botocore,
                        "botocore.exceptions": _exceptions})
    sys.modules["boto3"].client = lambda *a, **kw: None

import dispatcher

ISSUE = {"issue_number": 12, "title": "agent-x", "body": "", "base_branch": "main", "repo": "org/repo"}
FIX = {"mode": "fix", "repo": "org/repo", "pr_number": 25, "comment_id": 9}


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()
        self.client.invoke_agent_runtime.return_value = {"runtimeSessionId": "sid", "response": None}
        patches = [mock.patch.dict(os.environ, {"AGENT_RUNTIME_ARN": "arn:rt"}),
                   mock.patch.object(dispatcher.boto3, "client", return_value=self.client),
                   mock.patch.object(dispatcher, "mint_token", side_effect=lambda repo: f"tok-for-{repo}")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_session_ids_are_deterministic_unique_and_long_enough(self):
        a, b = dispatcher.session_id_for(ISSUE), dispatcher.session_id_for(dict(ISSUE))
        self.assertEqual(a, b)
        self.assertGreaterEqual(len(a), 33)
        self.assertNotEqual(a, dispatcher.session_id_for({**ISSUE, "issue_number": 13}))
        self.assertNotEqual(dispatcher.session_id_for(FIX), dispatcher.session_id_for({**FIX, "comment_id": 10}))

    def test_invokes_runtime_with_session_and_runtime_arn(self):
        resp = dispatcher.handler(ISSUE, None)
        self.assertEqual(resp["statusCode"], 200)
        kwargs = self.client.invoke_agent_runtime.call_args.kwargs
        self.assertEqual(kwargs["runtimeSessionId"], dispatcher.session_id_for({**ISSUE, "mode": "issue"}))
        self.assertEqual(json.loads(kwargs["payload"])["runtime_arn"], "arn:rt")

    def test_payload_carries_token_minted_for_event_repo(self):
        dispatcher.handler({**ISSUE, "repo": "org/other"}, None)
        dispatcher.mint_token.assert_called_once_with("org/other")
        payload = json.loads(self.client.invoke_agent_runtime.call_args.kwargs["payload"])
        self.assertEqual(payload["github_token"], "tok-for-org/other")

    def test_mint_failure_returns_502_without_invoking(self):
        dispatcher.mint_token.side_effect = RuntimeError("secret-key-material")
        resp = dispatcher.handler(ISSUE, None)
        self.assertEqual(resp["statusCode"], 502)
        self.assertEqual(json.loads(resp["body"]), {"error": "could not mint GitHub token"})
        self.client.invoke_agent_runtime.assert_not_called()

    def test_mint_failure_logs_only_exception_class(self):
        dispatcher.mint_token.side_effect = ImportError("secret-key-material")
        with self.assertLogs(level="ERROR") as logs:
            resp = dispatcher.handler(ISSUE, None)
        output = "\n".join(logs.output)
        self.assertIn("ImportError", output)
        self.assertNotIn("secret-key-material", output)
        self.assertEqual(resp["statusCode"], 502)
        self.assertEqual(json.loads(resp["body"]), {"error": "could not mint GitHub token"})

    def test_token_never_in_response(self):
        resp = dispatcher.handler(ISSUE, None)
        self.assertEqual(resp["statusCode"], 200)
        self.assertNotIn("tok-for-org/repo", resp["body"])

    def test_gates(self):
        self.assertEqual(dispatcher.handler({**ISSUE, "title": "not an agent issue"}, None)["statusCode"], 400)
        self.assertEqual(dispatcher.handler({"mode": "fix", "repo": "r"}, None)["statusCode"], 400)
        self.assertEqual(dispatcher.handler({**ISSUE, "mode": "other"}, None)["statusCode"], 400)
        self.client.invoke_agent_runtime.assert_not_called()
        self.assertEqual(dispatcher.handler(FIX, None)["statusCode"], 200)


if __name__ == "__main__":
    unittest.main()
