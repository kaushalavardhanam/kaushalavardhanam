#!/bin/sh
# Fetch the AgentCore worker git token and install the HTTPS credential helper.
# Use this on images that do not yet call resolve_git_token() at adapter boot.
# Safe to re-run. Never prints the token.
set -eu

region="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
secret_id="${CURSOR_GIT_TOKEN_SECRET_ID:-${CURSOR_GIT_TOKEN_SECRET_NAME:-cursor-agentcore-worker-git-token}}"

if [ -z "${CURSOR_GIT_TOKEN:-}${GH_TOKEN:-}${GH_WORKER_TOKEN:-}" ]; then
  raw="$(aws secretsmanager get-secret-value \
    --region "$region" \
    --secret-id "$secret_id" \
    --query SecretString \
    --output text)"
  CURSOR_GIT_TOKEN="$(printf '%s' "$raw" | python3 -c '
import json, sys
raw = sys.stdin.read().strip()
if not raw or raw == "None":
    raise SystemExit("empty secret")
if raw.startswith("{"):
    data = json.loads(raw)
    for key in ("CURSOR_GIT_TOKEN", "GH_TOKEN", "GH_WORKER_TOKEN", "GITHUB_TOKEN", "token"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            print(value.strip(), end="")
            raise SystemExit(0)
    raise SystemExit("secret JSON has no token key")
print(raw, end="")
')"
  export CURSOR_GIT_TOKEN
fi

root="$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)"
"$root/.github/scripts/configure-worker-git.sh"
echo "load-worker-git-token: credential helper installed"
