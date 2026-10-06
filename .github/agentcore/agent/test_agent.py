"""Tests for the runtime entrypoint's accept / de-dup / session-stop contract.

    python -m unittest test_agent
"""

from __future__ import annotations

import sys
import threading
import types
import unittest
from unittest import mock

import agent

ISSUE = {"issue_number": 12, "title": "agent-x", "body": "", "base_branch": "main",
         "repo": "org/repo", "runtime_arn": "arn:rt"}


class HandleTests(unittest.TestCase):
    def setUp(self):
        agent._running.clear()
        self.events = []

    def test_returns_accepted_immediately_and_stops_session_after(self):
        release = threading.Event()
        done = threading.Event()

        def job(payload):
            release.wait(5)
            return {"status": "ok"}

        def stop(arn, sid):
            self.events.append(("stop", arn, sid))
            done.set()

        resp = agent.handle(ISSUE, "sess-1", add_task=lambda n: self.events.append(("add", n)) or 1,
                            complete_task=lambda t: self.events.append(("complete", t)), job=job, stop=stop)
        self.assertEqual(resp, {"status": "accepted", "job": "issue:org/repo#12"})
        self.assertNotIn("stop", [e[0] for e in self.events])  # still running

        dup = agent.handle(ISSUE, "sess-1", job=job, stop=stop)
        self.assertEqual(dup["status"], "already_running")

        release.set()
        self.assertTrue(done.wait(5))
        self.assertEqual(self.events, [("add", "issue:org/repo#12"), ("complete", 1), ("stop", "arn:rt", "sess-1")])

    def test_failed_job_still_completes_task_and_stops(self):
        def job(payload):
            raise RuntimeError("boom")

        agent.handle({**ISSUE, "mode": "fix", "pr_number": 25, "comment_id": 9}, "s",
                     complete_task=lambda t: self.events.append("complete"), job=job,
                     stop=lambda a, s: self.events.append("stop"), background=False)
        self.assertEqual(self.events, ["complete", "stop"])
        self.assertEqual(agent._running, set())

    def test_validation(self):
        with self.assertRaises(ValueError):
            agent.handle({"mode": "fix", "repo": "org/repo"}, "s", job=lambda p: {})
        with self.assertRaises(ValueError):
            agent.handle({"mode": "nope", "repo": "r"}, "s", job=lambda p: {})


class RunJobTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        fake = types.ModuleType("jobs")
        fake.run_issue_job = lambda p, get_token: self.calls.append(("issue", get_token())) or {"ok": 1}
        fake.run_fix_job = lambda p, get_token: self.calls.append(("fix", get_token())) or {"ok": 2}
        patcher = mock.patch.dict(sys.modules, {"jobs": fake})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_issue_job_gets_payload_token(self):
        self.assertEqual(agent.run_job({**ISSUE, "mode": "issue", "github_token": "tok-1"}), {"ok": 1})
        self.assertEqual(self.calls, [("issue", "tok-1")])

    def test_fix_job_gets_payload_token(self):
        agent.run_job({"mode": "fix", "repo": "org/repo", "github_token": "tok-2"})
        self.assertEqual(self.calls, [("fix", "tok-2")])

    def test_missing_token_raises_and_runs_nothing(self):
        for payload in ({**ISSUE, "mode": "issue"}, {**ISSUE, "mode": "issue", "github_token": ""}):
            with self.assertRaisesRegex(ValueError, "github_token"):
                agent.run_job(payload)
        self.assertEqual(self.calls, [])

    def test_token_not_in_handle_response(self):
        agent._running.clear()
        resp = agent.handle({**ISSUE, "github_token": "secret-tok"}, "s", job=lambda p: {},
                            stop=lambda a, s: None, background=False)
        self.assertNotIn("secret-tok", str(resp))


if __name__ == "__main__":
    unittest.main()
