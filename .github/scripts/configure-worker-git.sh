#!/bin/sh
# Configure HTTPS git push for this checkout using a worker env token.
# Safe to run on every AgentCore session start. No-op if no token is set.
set -eu
root="$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)"
helper="$root/.github/scripts/git-credential-github-env"
chmod +x "$helper"

token="${CURSOR_GIT_TOKEN:-${GH_TOKEN:-${GH_WORKER_TOKEN:-${GITHUB_TOKEN:-}}}}"
if [ -z "$token" ]; then
  echo "configure-worker-git: no CURSOR_GIT_TOKEN/GH_TOKEN/GH_WORKER_TOKEN — push/PR will fail" >&2
  exit 1
fi

git config --global credential.https://github.com.helper ""
git config --global --replace-all credential.helper "!$helper"
git config --global url.https://github.com/.insteadOf git@github.com:
echo "configure-worker-git: credential helper installed for github.com"
