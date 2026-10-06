"""Static checks that only the dispatcher can read the GitHub App secret."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

MAIN_TF = Path(__file__).resolve().parent.parent / "main.tf"


def resource_block(text: str, rtype: str, name: str) -> str:
    """Return the body of `resource "rtype" "name" { ... }` by brace matching."""
    m = re.search(rf'resource\s+"{rtype}"\s+"{name}"\s*\{{', text)
    assert m, f"{rtype}.{name} not found"
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.start():i]


class TerraformPolicyTest(unittest.TestCase):
    def setUp(self):
        self.text = MAIN_TF.read_text()

    def test_get_secret_value_only_in_dispatcher_policy(self):
        dispatcher = resource_block(
            self.text, "aws_iam_role_policy", "dispatcher_lambda_github_app_secret"
        )
        self.assertIn("secretsmanager:GetSecretValue", dispatcher)
        self.assertIn("aws_iam_role.dispatcher_lambda.id", dispatcher)
        self.assertIn("aws_secretsmanager_secret.github_app.arn", dispatcher)
        remainder = self.text.replace(dispatcher, "")
        self.assertNotIn("GetSecretValue", remainder)

    def test_runtime_has_no_secret_access(self):
        runtime = resource_block(
            self.text, "aws_bedrockagentcore_agent_runtime", "decomposer"
        )
        self.assertNotIn("GITHUB_APP_SECRET_ARN", runtime)
        self.assertNotIn("secretsmanager", runtime)

    def test_dispatcher_gets_secret_arn_env(self):
        fn = resource_block(self.text, "aws_lambda_function", "dispatcher")
        self.assertIn("GITHUB_APP_SECRET_ARN", fn)

    def test_dispatcher_packages_github_app(self):
        archive = re.search(
            r'data\s+"archive_file"\s+"dispatcher_stub"\s*\{.*?\n\}', self.text, re.S
        )
        self.assertTrue(archive)
        self.assertIn("github_app.py", archive.group(0))


if __name__ == "__main__":
    unittest.main()
