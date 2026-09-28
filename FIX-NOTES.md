# Fix notes: Claude Bedrock agent never opens a PR

The `agent-*` → In Progress backend dispatches `.github/workflows/claude-agent-issue.yml`,
which runs `anthropics/claude-code-action@v1` on Amazon Bedrock (OIDC-federated
IAM role, no static credentials). The agent was supposed to branch, commit, and
open a PR, but **no `claude/*` branch or PR was ever created**. Two independent
failures were found in the run logs.

---

## Failure 1 — OIDC AssumeRole rejected (AWS-side, NOT fixed in this branch)

**Symptom** (e.g. run `36334703239`, ~status=failure after retries):

```
##[error]Could not assume role with OIDC: Not authorized to perform sts:AssumeRoleWithWebIdentity
```

The `Configure AWS credentials` step (`aws-actions/configure-aws-credentials@v4`)
cannot assume `arn:aws:iam::146666888814:role/github-actions-claude-bedrock`.
When this fails the job dies before Claude ever starts, so nothing is produced.

**Root cause:** the trust policy on `github-actions-claude-bedrock` does not
accept the GitHub OIDC token for this repo — the OIDC provider is missing, the
`sub` condition doesn't match `repo:kaushalavardhanam/kaushalavardhanam:*`, or
the audience isn't `sts.amazonaws.com`.

**Why it isn't fixed here:** this is an IAM change in AWS account `146666888814`.
This host has no AWS credentials, so I did not (and must not) touch AWS. The user
must apply the trust policy below.

### Remediation (user action, AWS Console/CLI)

1. Confirm the OIDC provider exists in the account (once per account):
   - Provider URL: `https://token.actions.githubusercontent.com`
   - Audience (client ID): `sts.amazonaws.com`
   - (IAM → Identity providers → Add provider → OpenID Connect, if absent.)

2. Set this **trust policy** on role `github-actions-claude-bedrock`. Replace
   `146666888814` if the account ever differs; it already matches the repo var
   `AWS_ACCOUNT_ID`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::146666888814:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:kaushalavardhanam/kaushalavardhanam:*"
        }
      }
    }
  ]
}
```

Notes:
- `aud` uses `StringEquals` and MUST be `sts.amazonaws.com` — that is the
  audience `aws-actions/configure-aws-credentials@v4` requests.
- `sub` uses `StringLike` with the `:*` wildcard so it matches every ref/branch/
  environment for this repo (`repo:OWNER/REPO:ref:refs/heads/main`,
  `...:environment:prod`, etc.). Tighten it later if you want to restrict to a
  branch, e.g. `repo:kaushalavardhanam/kaushalavardhanam:ref:refs/heads/main`.
- The workflow sets `role-skip-session-tagging: true`, so the trust policy does
  **not** need to allow `sts:TagSession`. Keep that workflow setting, or add a
  `sts:TagSession` statement if you remove it.

Apply via CLI:

```bash
aws iam update-assume-role-policy \
  --role-name github-actions-claude-bedrock \
  --policy-document file://trust-policy.json
```

(This is the trust/assume-role policy only. The separate `BedrockInvokeClaude`
inline permissions policy documented in the workflow header is unchanged.)

---

## Failure 2 — Claude's GitHub tools denied (the main bug; FIXED in this branch)

**Symptom** (e.g. run `36367248109`, status=success but produced nothing). In
the run log, after Bedrock auth succeeds and Claude starts:

```
OVERRIDE_GITHUB_TOKEN:            (empty)
  "github_token": "",
Using GITHUB_TOKEN from OIDC
  "subtype": "permission_denied",
        "type": "tool_result",
```

`permission_denied` repeats on every GitHub write tool call, so
create-branch / `create_pull_request` never succeed and the run ends "green"
having created nothing. Confirmed: `gh pr list` shows only human PRs and
Cursor's bot PRs; there is no `claude/*` branch or PR.

**Root cause:** the workflow passed **no `github_token`** to
`anthropics/claude-code-action@v1`. In `action.yml`, the `github_token` input
feeds `OVERRIDE_GITHUB_TOKEN`; when it is empty and no official Claude GitHub
App is installed (we're on the Bedrock path, so it isn't), the action mints a
token from the workflow's **OIDC identity** ("Using GITHUB_TOKEN from OIDC").
That token is **not** the job's `GITHUB_TOKEN` and does **not** inherit the
job's `permissions:` block, so the action's internal GitHub MCP server rejects
Claude's branch/PR tool calls — even though the job grants `contents: write` and
`pull-requests: write`. The job-level permissions were never the problem; the
token the MCP server actually held was.

**Fix applied** in `.github/workflows/claude-agent-issue.yml`, on the
`Run Claude Code` step:

```yaml
      - name: Run Claude Code
        uses: anthropics/claude-code-action@v1
        with:
          use_bedrock: "true"
          github_token: ${{ secrets.GITHUB_TOKEN }}   # <-- added
          display_report: "true"
          show_full_output: "true"
          base_branch: ${{ github.event.inputs.base_branch }}
          claude_args: "--model global.anthropic.claude-sonnet-4-5-20250929-v1:0 --verbose"
          prompt: | ...
```

Passing the job's `GITHUB_TOKEN` explicitly gives the action's GitHub MCP server
a token that inherits the job's `permissions:` (`contents: write` +
`pull-requests: write` + `issues: write`), which authorizes the create-branch,
commit, and `create_pull_request` tools.

Verified against the action's own `action.yml` (input `github_token` →
`OVERRIDE_GITHUB_TOKEN`) and `docs/usage.md` / `docs/setup.md`: for non-App auth
(Bedrock/Vertex/Foundry), an explicit `github_token` is the supported way to
grant Claude's GitHub tools. Branch/commit/PR are built-in GitHub tools of the
action, so no extra `--allowedTools` entry is required to enable them; the job's
`permissions:` block already carries `contents: write` and `pull-requests: write`
(unchanged).

### Known limitation of this fix (GitHub platform behavior)

A pull request opened using the default `GITHUB_TOKEN` does **not** trigger other
workflows (GitHub suppresses recursive workflow runs from the automatic token by
design). The PR is created and fully functional, but any `on: pull_request` CI
will **not** auto-run on it. If you need the agent's PR to trigger downstream
CI, replace `github_token` with a token from a **GitHub App**
(`actions/create-github-app-token`, Contents + Pull requests write) or a
**fine-grained PAT** with the same scopes, stored as a secret. That is a
follow-up choice, not required to make PR creation itself work.

---

## Verification status

- Failure 2 fix: workflow edited on branch `fix/claude-agent-pr-creation`; YAML
  validated (parses; `github_token` wired; job permissions confirmed).
- End-to-end proof (a live `agent-*` dispatch producing a `claude/*` branch and
  PR) is **gated on Failure 1**: until the OIDC trust policy above is applied,
  the job dies at the AWS step before Claude runs. Once the trust policy is in
  place, dispatch `claude-agent-issue.yml` on a test `agent-*` issue and confirm
  a `claude/*` branch + PR appear.
