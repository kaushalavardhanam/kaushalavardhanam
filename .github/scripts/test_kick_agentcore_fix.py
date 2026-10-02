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


COMMENTS_URL = f"{API}/issues/25/comments?sort=created&per_page=100&page=1"
OPEN_PRS_URL = f"{API}/pulls?state=open&per_page=100&page=1"


def fix_comment(cid, login="taufiq", body="/agent fix\nimport is wrong", type_="User"):
    return {"id": cid, "body": body, "user": {"login": login, "type": type_}}


class FakeGitHub:
    def __init__(self, *, head_ref="agentcore/x-12", head_repo=REPO, issue_author="taufiq",
                 permission="read", existing_comments=(), other_prs=()):
        self.posts = []
        pr = {"number": 25, "state": "open", "body": "Closes #12",
              "head": {"ref": head_ref, "repo": {"full_name": head_repo}}}
        self.routes = {
            f"{API}/pulls/25": pr,
            OPEN_PRS_URL: [pr, *other_prs],
            f"{API}/issues/12": {"user": {"login": issue_author}},
            COMMENTS_URL: list(existing_comments),
        }
        self.permission = permission

    def __call__(self, url, token, payload=None, method=None):
        if payload is not None:
            self.posts.append((url, payload))
            return {}
        if "/permission" in url:
            return {"permission": self.permission}
        return self.routes[url]


def run_main(gh, user="taufiq", body="/agent fix\r\nimport is wrong", environ=None, dispatch=None,
             expect=0):
    if environ is None:
        environ = {"PR_NUMBER": "25", "COMMENT_ID": "99", "COMMENT_AUTHOR": user,
                   "COMMENT_BODY": body}
    environ = {"GITHUB_REPOSITORY": REPO, "GITHUB_TOKEN": "t", **environ}
    dispatch = dispatch or mock.Mock(return_value={"status_code": 202, "response": {}})
    with mock.patch.dict(os.environ, environ, clear=True), \
            mock.patch.object(kick, "github_request", gh), \
            mock.patch.object(kick, "invoke_dispatcher", dispatch):
        assert kick.main() == expect
    return dispatch


def run_scan(gh, pr_number="", **kwargs):
    return run_main(gh, environ={"PR_NUMBER": pr_number}, **kwargs)


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
        self.assertIn("<!-- agentcore-fix-rejected:99 -->", gh.posts[-1][1]["body"])

    def test_non_agent_or_fork_pr_is_ignored(self):
        for kwargs in ({"head_ref": "feature/x"}, {"head_repo": "evil/fork"}):
            with self.subTest(**kwargs):
                gh = FakeGitHub(**kwargs)
                run_main(gh).assert_not_called()
                self.assertEqual(gh.posts, [])

    def test_already_dispatched_comment_is_skipped(self):
        gh = FakeGitHub(existing_comments=[{"body": "<!-- agentcore-fix-kickoff:99 -->"}])
        run_main(gh).assert_not_called()

    def test_scan_dispatches_only_unanswered_fix_comments(self):
        gh = FakeGitHub(existing_comments=[
            fix_comment(90),
            {"id": 91, "body": "<!-- agentcore-fix-kickoff:90 -->\nOn it", "user": {"type": "Bot"}},
            fix_comment(92, body="looks good"),
            fix_comment(93, type_="Bot"),
            fix_comment(94),
        ])
        dispatch = run_scan(gh)
        self.assertEqual([c.args[2]["comment_id"] for c in dispatch.call_args_list], [94])

    def test_scan_skips_non_agent_and_closed_prs(self):
        others = [
            {"number": 30, "state": "open", "head": {"ref": "feature/y", "repo": {"full_name": REPO}}},
            {"number": 31, "state": "open", "head": {"ref": "agentcore/z-9", "repo": {"full_name": "evil/fork"}}},
        ]
        gh = FakeGitHub(other_prs=others, existing_comments=[fix_comment(94)])
        dispatch = run_scan(gh)  # never asks for comments of PR 30/31 (would KeyError)
        dispatch.assert_called_once()

    def test_scan_does_not_repeat_a_rejection(self):
        gh = FakeGitHub(existing_comments=[
            fix_comment(95, login="drive-by"),
            {"id": 96, "body": "<!-- agentcore-fix-rejected:95 -->\n@drive-by ...", "user": {"type": "Bot"}},
        ])
        run_scan(gh).assert_not_called()
        self.assertEqual(gh.posts, [])

    def test_scan_limited_to_one_pr(self):
        gh = FakeGitHub(existing_comments=[fix_comment(94)])
        del gh.routes[OPEN_PRS_URL]  # must not list all PRs
        run_scan(gh, pr_number="25").assert_called_once()

    def test_scan_failure_withdraws_claim_and_exits_nonzero(self):
        class ClaimingGitHub(FakeGitHub):
            def __call__(self, url, token, payload=None, method=None):
                if method == "DELETE":
                    self.posts.append((url, "DELETE"))
                    return {}
                if payload is not None and url.endswith("/issues/25/comments"):
                    self.posts.append((url, payload))
                    return {"id": 500}
                return super().__call__(url, token, payload, method)

        gh = ClaimingGitHub(existing_comments=[fix_comment(94)])
        run_scan(gh, dispatch=mock.Mock(side_effect=RuntimeError("throttled")), expect=1)
        self.assertIn((f"{API}/issues/comments/500", "DELETE"), gh.posts)


if __name__ == "__main__":
    unittest.main()
