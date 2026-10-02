"""Tests for the runtime entrypoint's accept / de-dup / session-stop contract.

    python -m unittest test_agent
"""

from __future__ import annotations

import threading
import unittest

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


if __name__ == "__main__":
    unittest.main()
