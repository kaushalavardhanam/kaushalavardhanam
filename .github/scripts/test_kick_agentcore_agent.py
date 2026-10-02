"""Unit tests for the AgentCore kickoff script's dispatcher invoke.

Runs offline: a fake ``lambda`` client is injected via ``boto3.client`` so no
AWS call is made. Exercises the async (fire-and-forget) invoke contract added
to fix the CI 60s ``ReadTimeoutError`` — the dispatcher fans out to a
many-minutes AgentCore run, so the kickoff must NOT block on it.

    python -m unittest test_kick_agentcore_agent
"""

from __future__ import annotations

import unittest
import unittest.mock

import kick_agentcore_agent
from kick_agentcore_agent import invoke_dispatcher


class _FakeLambdaClient:
    """Captures the invoke() request and returns a canned response."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.last_kwargs: dict = {}
        self.call_count = 0

    def invoke(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        return self._response


class _NoReadPayload:
    """A Payload sentinel that fails the test if .read() is ever called.

    An async ("Event") invoke returns no function result, so the code must not
    attempt to read a response body on the success path.
    """

    def read(self):  # pragma: no cover - must never be reached on success
        raise AssertionError("invoke_dispatcher must not read the Payload on an async 202")


class InvokeDispatcherAsyncTest(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_boto3_client = kick_agentcore_agent.boto3.client

    def tearDown(self) -> None:
        kick_agentcore_agent.boto3.client = self._orig_boto3_client

    def _install_fake(self, response: dict) -> _FakeLambdaClient:
        fake = _FakeLambdaClient(response)

        def _factory(service_name, *args, **kwargs):
            assert service_name == "lambda", service_name
            return fake

        kick_agentcore_agent.boto3.client = _factory
        return fake

    def test_uses_event_invocation_type(self) -> None:
        """The invoke must be async: InvocationType='Event'."""
        fake = self._install_fake({"StatusCode": 202, "Payload": _NoReadPayload()})
        payload = {"issue_number": 42, "title": "agent-x", "body": "b",
                   "base_branch": "main", "repo": "o/r"}

        result = invoke_dispatcher("agentcore-decomposer-dispatcher", "us-east-1", payload)

        self.assertEqual(fake.call_count, 1)
        self.assertEqual(fake.last_kwargs["InvocationType"], "Event")
        self.assertEqual(fake.last_kwargs["FunctionName"], "agentcore-decomposer-dispatcher")
        # Payload is still forwarded so the dispatcher receives the issue.
        self.assertIn(b'"issue_number": 42', fake.last_kwargs["Payload"])

    def test_tolerates_202_with_empty_payload(self) -> None:
        """A 202 with an empty/absent function result is success, not an error.

        The kickoff-comment code in start_agent reads result['status_code'] and
        result['response'], so both keys must be present.
        """
        # No 'Payload' key at all — the async response has no body to read.
        self._install_fake({"StatusCode": 202})
        payload = {"issue_number": 7, "title": "agent-y", "body": "",
                   "base_branch": "main", "repo": "o/r"}

        result = invoke_dispatcher("fn", "us-east-1", payload)

        self.assertEqual(result["status_code"], 202)
        self.assertEqual(result["response"], {})

    def test_does_not_read_payload_on_success(self) -> None:
        """Even when a Payload object is present, success must not .read() it."""
        self._install_fake({"StatusCode": 202, "Payload": _NoReadPayload()})
        # _NoReadPayload.read() raises if touched; a clean return proves it wasn't.
        result = invoke_dispatcher("fn", "us-east-1",
                                   {"issue_number": 1, "title": "agent-z", "body": "",
                                    "base_branch": "main", "repo": "o/r"})
        self.assertEqual(result["status_code"], 202)

    def test_non_202_status_raises(self) -> None:
        """A non-202 StatusCode is a dispatch failure and must raise."""
        self._install_fake({"StatusCode": 500})
        with self.assertRaises(RuntimeError):
            invoke_dispatcher("fn", "us-east-1",
                              {"issue_number": 1, "title": "agent-z", "body": "",
                               "base_branch": "main", "repo": "o/r"})


if __name__ == "__main__":
    unittest.main()


class StartAgentClaimTests(unittest.TestCase):
    """The kickoff comment is the dedup key, so it is posted BEFORE dispatching."""

    def setUp(self):
        self.calls = []

        def fake_github(url, token, payload=None, method=None):
            self.calls.append((method or ("POST" if payload else "GET"), url))
            if url.endswith("/comments?per_page=100"):
                return []
            return {"id": 77} if payload else {}

        patcher = unittest.mock.patch.object(kick_agentcore_agent, "github_request", fake_github)
        patcher.start()
        self.addCleanup(patcher.stop)

    def start(self, issue, dispatch):
        with unittest.mock.patch.object(kick_agentcore_agent, "invoke_dispatcher", dispatch):
            kick_agentcore_agent.start_agent(issue, "org/repo", "t", "fn", "us-east-1", "", "main")

    def issue(self, **kw):
        return {"number": 12, "title": "agent-x", "state": "open", "labels": [], **kw}

    def test_claim_is_posted_before_dispatch(self):
        def dispatch(fn, region, payload):
            self.assertIn(("POST", "https://api.github.com/repos/org/repo/issues/12/comments"), self.calls)
            self.assertEqual(payload["mode"], "issue")
            return {"status_code": 202}

        self.start(self.issue(), dispatch)

    def test_claim_is_withdrawn_when_dispatch_fails(self):
        dispatch = unittest.mock.Mock(side_effect=RuntimeError("throttled"))
        with self.assertRaises(RuntimeError):
            self.start(self.issue(), dispatch)
        self.assertEqual(self.calls[-1], ("DELETE", "https://api.github.com/repos/org/repo/issues/comments/77"))

    def test_closed_issue_is_not_dispatched(self):
        dispatch = unittest.mock.Mock()
        self.start(self.issue(state="closed"), dispatch)
        dispatch.assert_not_called()
