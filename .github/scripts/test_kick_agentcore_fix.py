"""Unit tests for the ``/agent fix`` kickoff script.

Offline: ``github_request`` and ``invoke_dispatcher`` are patched, so no GitHub
or AWS call is made.

    python -m unittest test_kick_agentcore_fix
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import kick_agentcore_fix as kick

REPO = "org/repo"
API = f"{kick.GITHUB_API}/repos/{REPO}"


class FakeGitHub:
    def __init__(self, *, head_ref="agentcore/x-12", head_repo=REPO, issue_author="taufiq",
                 permission="read", existing_comments=()):
        self.posts = []
        self.routes = {
            f"{API}/pulls/25": {"state": "open", "body": "Closes #12",
                                "head": {"ref": head_ref, "repo": {"full_name": head_repo}}},
            f"{API}/issues/12": {"user": {"login": issue_author}},
            f"{API}/issues/25/comments?per_page=100": list(existing_comments),
        }
        self.permission = permission

    def __call__(self, url, token, payload=None, method=None):
        if payload is not None:
            self.posts.append((url, payload))
            return {}
        if "/permission" in url:
            return {"permission": self.permission}
        return self.routes[url]


def run_main(gh, user="taufiq", body="/agent fix\r\nimport is wrong"):
    environ = {"GITHUB_REPOSITORY": REPO, "GITHUB_TOKEN": "t", "PR_NUMBER": "25",
               "COMMENT_ID": "99", "COMMENT_AUTHOR": user, "COMMENT_BODY": body}
    dispatch = mock.Mock(return_value={"status_code": 202, "response": {}})
    with mock.patch.dict(os.environ, environ), \
            mock.patch.object(kick, "github_request", gh), \
            mock.patch.object(kick, "invoke_dispatcher", dispatch):
        assert kick.main() == 0
    return dispatch


class KickFixTests(unittest.TestCase):
    def test_is_fix_command(self):
        self.assertTrue(kick.is_fix_command("/agent fix\r\ndetails"))
        self.assertTrue(kick.is_fix_command("  /Agent Fix please"))
        self.assertFalse(kick.is_fix_command("looks good\n/agent fix"))
        self.assertFalse(kick.is_fix_command(""))

    def test_issue_author_dispatches_fix_and_acks(self):
        gh = FakeGitHub()
        dispatch = run_main(gh)
        payload = dispatch.call_args.args[2]
        self.assertEqual(payload, {"mode": "fix", "repo": REPO, "pr_number": 25, "comment_id": 99})
        urls = [u for u, _ in gh.posts]
        self.assertIn(f"{API}/issues/comments/99/reactions", urls)
        ack = [p["body"] for u, p in gh.posts if u.endswith("/issues/25/comments")][0]
        self.assertIn("<!-- agentcore-fix-kickoff:99 -->", ack)

    def test_write_collaborator_may_dispatch(self):
        dispatch = run_main(FakeGitHub(permission="write"), user="maintainer")
        dispatch.assert_called_once()

    def test_untrusted_commenter_is_rejected(self):
        gh = FakeGitHub(permission="read")
        dispatch = run_main(gh, user="drive-by")
        dispatch.assert_not_called()
        self.assertIn("can only be used", gh.posts[-1][1]["body"])

    def test_non_agent_or_fork_pr_is_ignored(self):
        for kwargs in ({"head_ref": "feature/x"}, {"head_repo": "evil/fork"}):
            with self.subTest(**kwargs):
                gh = FakeGitHub(**kwargs)
                run_main(gh).assert_not_called()
                self.assertEqual(gh.posts, [])

    def test_already_dispatched_comment_is_skipped(self):
        gh = FakeGitHub(existing_comments=[{"body": "<!-- agentcore-fix-kickoff:99 -->"}])
        run_main(gh).assert_not_called()


if __name__ == "__main__":
    unittest.main()
