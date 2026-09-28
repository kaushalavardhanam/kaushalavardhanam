# AgentCore decomposer backend — Terraform

AWS-native IaC for the third agent backend of this repo. Where the Claude and
Cursor backends run the agent on hosted CI, this backend runs **all compute and
LLM inference on AWS**:

```
board -> "In Progress" -> GH Actions kick -> OIDC role -> dispatcher Lambda
      -> Bedrock AgentCore Runtime agent -> decompose issue into sub-tasks
      -> implement via Bedrock -> open PR
```

This directory provisions the AWS side only. The **agent container image** and
the **real dispatcher handler** are separate work items; here they are a
placeholder image URI (`var.agent_container_image_uri`) and a minimal Python
stub (`lambda/dispatcher.py`).

Authentication from GitHub Actions is via **OIDC — no static AWS keys**,
mirroring the existing Claude backend (`.github/workflows/claude-agent-issue.yml`).

## What `terraform apply` creates (region `us-east-1` by default)

1. **Bedrock AgentCore Runtime** (`aws_bedrockagentcore_agent_runtime`) hosting
   the containerized agent, plus its **execution role**. The execution role is
   granted `bedrock:InvokeModel` / `InvokeModelWithResponseStream` on the same
   three targets the Claude backend uses for global CRIS:
   - the regional inference-profile ARN,
   - the regional `anthropic.*` foundation-model ARN,
   - the **global** `anthropic.*` foundation-model ARN (no region in the ARN),
     gated by `aws:RequestedRegion = "unspecified"`.
   It can also read the GitHub App secret and write CloudWatch Logs.
2. **Dispatcher Lambda** (`python3.12`) that receives
   `{issue_number, title, body, base_branch, repo}` and invokes the AgentCore
   Runtime agent, plus its role (logs + `bedrock-agentcore:InvokeAgentRuntime`
   on the runtime). The handler is a stub for now.
3. **OIDC IAM role** for GitHub Actions
   (`sts:AssumeRoleWithWebIdentity` on `token.actions.githubusercontent.com`,
   trust scoped to `repo:<github_repo>:*`), allowing `lambda:InvokeFunction` on
   the dispatcher. The **OIDC provider is reused** via a data source — it is not
   recreated (the Claude backend already created it).
4. **Secrets Manager secret** to hold the GitHub App private key + app/
   installation IDs. Created **empty** by default; populate out-of-band.

## Variables

| Variable | Default | Purpose |
|---|---|---|
| `aws_region` | `us-east-1` | Region for all resources. |
| `aws_account_id` | *(required)* | 12-digit account id; builds Bedrock + OIDC ARNs. Not a secret, just not hardcoded. |
| `name_prefix` | `agentcore-decomposer` | Prefix for created resource names. |
| `github_repo` | `kaushalavardhanam/kaushalavardhanam` | Repo allowed to assume the OIDC role. |
| `bedrock_model_id` | `global.anthropic.claude-sonnet-4-5-20250929-v1:0` | Model / inference profile the agent invokes. |
| `agent_container_image_uri` | placeholder ECR URI | Image for the AgentCore Runtime (separate work item). |
| `populate_github_app_secret` | `false` | If true, seeds the secret from `github_app_secret_value`. |
| `github_app_secret_value` | `""` (sensitive) | GitHub App JSON; only used when the flag above is true. Pass via uncommitted tfvars/env — never hardcode. |
| `tags` | see `variables.tf` | Tags on all resources. |

## Usage

```bash
terraform init
terraform plan  -var 'aws_account_id=123456789012'
terraform apply -var 'aws_account_id=123456789012'
```

To validate the config without any AWS credentials or backend:

```bash
terraform init -backend=false && terraform validate
```

> On some sandboxed hosts, `terraform init` fails to unzip providers if the
> system temp dir resolves to a denied path. If you hit
> `failed to compute checksum ... operation not permitted`, set
> `TMPDIR` to an allowed directory first (e.g. `export TMPDIR=/tmp/tf && mkdir -p "$TMPDIR"`).
> This is an environment quirk, not a config issue.

## Manual step: Bedrock model access

Terraform grants the IAM permissions to invoke the model, but **model access
must be granted in the Bedrock console** ("Model access") for `us-east-1` in the
target account. Without it, invocations fail with a model-access error even
though IAM allows the call — same requirement as the Claude backend.

## Resource-path note (AgentCore Runtime)

AgentCore Runtime resource types are new/evolving. A **first-class native
resource exists** in the `hashicorp/aws` provider (v6 line):
`aws_bedrockagentcore_agent_runtime`. This config uses it directly — no
`awscc_*` (Cloud Control) or `null_resource` + `local-exec` fallback was
needed. (For reference, `awscc_bedrockagentcore_runtime` also exists as an
alternative; the native `aws_` resource was chosen for a cleaner schema and
consistency with the rest of the repo's `aws_*` usage.)
