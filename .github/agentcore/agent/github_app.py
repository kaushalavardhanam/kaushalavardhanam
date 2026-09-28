"""GitHub App authentication for the AgentCore decomposer agent.

The agent opens its PR as a GitHub App installation, never with a hardcoded
token. The flow is:

1. Read the GitHub App credentials JSON from AWS Secrets Manager. The secret
   ARN arrives as the ``GITHUB_APP_SECRET_ARN`` environment variable, which the
   Terraform (``.github/agentcore/terraform/main.tf``) sets on the runtime and
   whose read the execution role is granted (``ReadGitHubAppSecret`` policy).
2. Mint a short-lived RS256 JWT signed with the App's private key
   (``iss = app_id``, 10-minute expiry) — the App-authentication step of the
   GitHub App API.
3. Exchange the JWT for a 1-hour *installation access token* via
   ``POST /app/installations/{installation_id}/access_tokens``.

The secret is expected to be a JSON string with these keys::

    {
        "app_id": "123456",
        "installation_id": "654321",
        "private_key": "<PEM private key text (BEGIN/END PRIVATE KEY block)>"
    }

``private_key`` may be a PEM string (with real or escaped newlines) or base64.
No credential is ever logged or written to disk.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import boto3
import jwt
import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
JWT_TTL_SECONDS = 9 * 60  # GitHub allows up to 10 minutes; leave headroom.
REQUEST_TIMEOUT = 30


@dataclass
class GitHubAppCredentials:
    app_id: str
    installation_id: str
    private_key: str  # PEM


def _load_private_key(raw: str) -> str:
    """Normalise the private key into PEM text.

    Accepts a PEM string with real newlines, one with ``\\n`` escapes, or a
    base64-encoded PEM blob.
    """
    if "BEGIN" in raw and "PRIVATE KEY" in raw:
        # Already PEM; fix escaped newlines if present.
        return raw.replace("\\n", "\n")
    # Otherwise assume base64-encoded PEM.
    try:
        decoded = base64.b64decode(raw, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("private_key is neither PEM nor valid base64 PEM") from exc
    if "PRIVATE KEY" not in decoded:
        raise ValueError("decoded private_key does not look like a PEM key")
    return decoded


def load_credentials(secret_arn: Optional[str] = None, region: Optional[str] = None) -> GitHubAppCredentials:
    """Fetch and parse the GitHub App credentials from Secrets Manager."""
    secret_arn = secret_arn or os.environ.get("GITHUB_APP_SECRET_ARN")
    if not secret_arn:
        raise RuntimeError("GITHUB_APP_SECRET_ARN is not set")

    region = region or os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION", "us-east-1"
    )
    client = boto3.client("secretsmanager", region_name=region)
    logger.info("Fetching GitHub App secret from Secrets Manager")
    resp = client.get_secret_value(SecretId=secret_arn)

    raw = resp.get("SecretString")
    if raw is None and resp.get("SecretBinary") is not None:
        raw = base64.b64decode(resp["SecretBinary"]).decode("utf-8")
    if not raw:
        raise RuntimeError("GitHub App secret is empty")

    data = json.loads(raw)
    missing = [k for k in ("app_id", "installation_id", "private_key") if not data.get(k)]
    if missing:
        raise RuntimeError(f"GitHub App secret missing keys: {missing}")

    return GitHubAppCredentials(
        app_id=str(data["app_id"]),
        installation_id=str(data["installation_id"]),
        private_key=_load_private_key(str(data["private_key"])),
    )


def mint_app_jwt(creds: GitHubAppCredentials, now: Optional[int] = None) -> str:
    """Create a short-lived RS256 JWT authenticating AS the GitHub App."""
    now = int(now if now is not None else time.time())
    payload = {
        # Backdate iat by 60s to tolerate clock skew per GitHub's guidance.
        "iat": now - 60,
        "exp": now + JWT_TTL_SECONDS,
        "iss": creds.app_id,
    }
    token = jwt.encode(payload, creds.private_key, algorithm="RS256")
    # PyJWT >= 2 returns str; older returns bytes.
    return token.decode("utf-8") if isinstance(token, bytes) else token


def get_installation_token(creds: Optional[GitHubAppCredentials] = None) -> str:
    """Return a 1-hour GitHub App *installation* access token.

    This is the token used for git push and PR creation. It is scoped to the
    App's installation and expires in an hour — never hardcoded, never logged.
    """
    creds = creds or load_credentials()
    app_jwt = mint_app_jwt(creds)

    url = f"{GITHUB_API}/app/installations/{creds.installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {app_jwt}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    logger.info("Requesting installation access token")
    resp = requests.post(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 201:
        raise RuntimeError(
            f"Failed to mint installation token: {resp.status_code} {resp.text[:200]}"
        )
    token = resp.json().get("token")
    if not token:
        raise RuntimeError("Installation token response missing 'token'")
    return token
