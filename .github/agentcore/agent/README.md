# AgentCore decomposer agent (container payload)

The containerized agent that the **Bedrock AgentCore Runtime** runs. It is the
"agent container image" that the Terraform in `.github/agentcore/terraform/`
provisions a runtime for (via `var.agent_container_image_uri`).

## Pipeline

```
agent-* issue -> In Progress -> GH Actions -> dispatcher Lambda
  -> AgentCore Runtime (this container)
       1. receive {issue_number, title, body, base_branch, repo}
       2. decompose the issue into ordered sub-tasks   (Bedrock)
       3. implement each sub-task -> file writes         (Bedrock)
       4. clone repo, branch off base, commit, push, open PR ("Closes #<n>")
```

All inference is on AWS Bedrock. The decompose->implement state machine reuses
the LangGraph supervisor/router pattern from the repo's
`coding-agent/graph_workflow.py`.

## Files

| File | Purpose |
|---|---|
| `agent.py` | AgentCore Runtime entrypoint. Serves `POST /invocations` + `GET /ping` on :8080 (via the `bedrock-agentcore` SDK, with a stdlib HTTP fallback). Holds `process_invocation()`. |
| `decomposer.py` | LangGraph `decompose -> implement` workflow. Bedrock-backed. |
| `bedrock_client.py` | `bedrock-runtime.invoke_model` wrapper (Claude Messages API). |
| `github_app.py` | Reads the GitHub App secret from Secrets Manager, mints an RS256 JWT, exchanges it for an installation access token. |
| `git_pr.py` | Clone / branch / commit / push / open PR using the installation token. |
| `Dockerfile` | linux/arm64, python 3.12, installs deps + git. |
| `requirements.txt` | langgraph, boto3, PyJWT, cryptography, requests, bedrock-agentcore. |

## Runtime contract (from the Terraform)

The AgentCore Runtime passes these **environment variables** (see
`aws_bedrockagentcore_agent_runtime.decomposer` in `../terraform/main.tf`):

| Env var | Meaning |
|---|---|
| `BEDROCK_MODEL_ID` | Model / inference-profile id to invoke. |
| `GITHUB_APP_SECRET_ARN` | ARN of the Secrets Manager secret with the GitHub App creds. |
| `GITHUB_REPO` | Default `owner/name` (backstops a missing `repo` in the payload). |

The execution role already grants `bedrock:InvokeModel[WithResponseStream]`
(three-target global-CRIS policy), `secretsmanager:GetSecretValue` on the
GitHub App secret, and CloudWatch Logs.

### GitHub App secret shape

The secret (created empty by the Terraform, populated out-of-band) must be a
JSON string with three keys:

```json
{
  "app_id": "123456",
  "installation_id": "654321",
  "private_key": "<PEM private key text>"
}
```

`private_key` holds the GitHub App's RSA private key. It may be raw PEM (the
usual `BEGIN/END ... PRIVATE KEY` block), PEM with `\n` escapes, or
base64-encoded PEM. No token is ever hardcoded: the installation token is
minted at runtime and lives for one hour.

## Model id

The agent invokes **Claude Opus 5** via the global CRIS inference profile:

```
global.anthropic.claude-opus-5
```

This is the default in `bedrock_client.py` and is overridden at runtime by the
`BEDROCK_MODEL_ID` env var.

> **Terraform follow-up:** `../terraform/variables.tf` still defaults
> `bedrock_model_id` to `global.anthropic.claude-sonnet-4-5-20250929-v1:0`.
> That default (and the model-access grant) must be updated to
> `global.anthropic.claude-opus-5` so the runtime env matches this agent. The
> Terraform is a separate work item / owned by another worker — flagging here
> rather than editing it.

## Build & push to ECR

AgentCore Runtime requires **linux/arm64** images.

```bash
ACCOUNT=146666888814
REGION=us-east-1
REPO=agentcore-decomposer
URI="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO"

# One-time: create the ECR repo
aws ecr create-repository --repository-name "$REPO" --region "$REGION"

# Auth docker to ECR
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

# Build for arm64 and push
docker buildx build --platform linux/arm64 -t "$URI:latest" --push .
```

Then point the Terraform at the image:

```bash
cd ../terraform
terraform apply \
  -var 'aws_account_id=146666888814' \
  -var "agent_container_image_uri=$URI:latest"
```

## Local sanity checks

```bash
# Byte-compile every module
python -m py_compile *.py

# Structural Dockerfile lint (if hadolint is available)
hadolint Dockerfile

# Build the image (if docker + buildx are available)
docker buildx build --platform linux/arm64 -t agentcore-decomposer:local .
```

The runtime logic is isolated in `process_invocation(payload)` (in `agent.py`),
so it can be exercised directly with a fake Bedrock client without the SDK or
network.
