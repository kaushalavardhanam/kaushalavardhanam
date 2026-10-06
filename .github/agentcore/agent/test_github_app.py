"""Unit tests for GitHub App installation tokens (``github_app.py``).

Offline: ``requests.post`` and the JWT minting are patched, creds are fake.

    python -m pytest test_github_app.py
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

# Only the real network/crypto calls need these packages, and the tests patch them out.
for _name in ("boto3", "jwt", "requests"):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules["requests"], "post"):
    sys.modules["requests"].post = None

import github_app

CREDS = github_app.GitHubAppCredentials(app_id="1", installation_id="42", private_key="fake")


def response(status=201, body=None, text=""):
    resp = mock.Mock()
    resp.status_code = status
    resp.text = text
    resp.json.return_value = {"token": "ghs_x"} if body is None else body
    return resp


class InstallationTokenTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(github_app, "mint_app_jwt", return_value="jwt")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_repository_scopes_body_to_one_repo(self):
        with mock.patch.object(github_app.requests, "post", return_value=response()) as post:
            token = github_app.get_installation_token(CREDS, repository="octo/widgets")
        self.assertEqual(token, "ghs_x")
        self.assertTrue(post.call_args.args[0].endswith("/app/installations/42/access_tokens"))
        self.assertEqual(post.call_args.kwargs["json"], {
            "repositories": ["widgets"],
            "permissions": {"contents": "write", "pull_requests": "write", "issues": "write"},
        })

    def test_unscoped_call_sends_no_repositories(self):
        with mock.patch.object(github_app.requests, "post", return_value=response()) as post:
            github_app.get_installation_token(CREDS)
        self.assertNotIn("json", post.call_args.kwargs)
        self.assertNotIn("repositories", str(post.call_args.kwargs))

    def test_non_201_raises(self):
        bad = response(status=403, text="forbidden")
        with mock.patch.object(github_app.requests, "post", return_value=bad):
            with self.assertRaises(RuntimeError):
                github_app.get_installation_token(CREDS, repository="octo/widgets")


if __name__ == "__main__":
    unittest.main()
