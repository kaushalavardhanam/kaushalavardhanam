# AgentCore decomposer backend — Terraform

AWS-native IaC for the repo's primary agent backend: **all compute and LLM
inference run on AWS**, serverlessly — a runtime session exists only while an
`agent-*` issue or an `/agent fix` comment is being worked on:

```
board -> "In Progress" (or "/agent fix" on an agent PR) -> GH Actions kick
      -> OIDC role -> dispatcher Lambda -> Bedrock AgentCore Runtime session
      -> Claude Code (Agent SDK, on Bedrock) plans, implements, tests -> PR
```

See the root README for the end-to-end diagrams and `../agent/README.md` for
the runtime. This directory provisions the AWS side; the image is built by
CodeBuild (`codebuild.tf`, `../deploy.sh`).

Authentication from GitHub Actions is via **OIDC — no static AWS keys**.

## What `terraform apply` creates (region `us-east-1` by default)

1. **Bedrock AgentCore Runtime** (`aws_bedrockagentcore_agent_runtime`) hosting
   the containerized agent, plus its **execution role**. The execution role is
   granted `bedrock:InvokeModel` / `InvokeModelWithResponseStream` on the same
   three targets the Claude backend uses for global CRIS:
   - the regional inference-profile ARN,
   - the regional `anthropic.*` foundation-model ARN,
   - the **global** `anthropic.*` foundation-model ARN (no region in the ARN),
     gated by `aws:RequestedRegion = "unspecified"`.
   It can write CloudWatch Logs but has **no Secrets Manager access**.
2. **Dispatcher Lambda** (`python3.12`) that receives
   `{issue_number, title, body, base_branch, repo}`, mints a repo-scoped token
   and invokes the AgentCore Runtime agent, plus its role (logs,
   `bedrock-agentcore:InvokeAgentRuntime` on the runtime, and
   `secretsmanager:GetSecretValue` on the GitHub App secret).
3. **OIDC IAM role** for GitHub Actions
   (`sts:AssumeRoleWithWebIdentity` on `token.actions.githubusercontent.com`,
   trust scoped to `repo:<github_repo>:*`), allowing `lambda:InvokeFunction` on
   the dispatcher. The **OIDC provider is reused** via a data source — it is not
   recreated (the Claude backend already created it).
4. **Secrets Manager secret** to hold the GitHub App private key + app/
   installation IDs. Created **empty** by default; populate out-of-band.

## GitHub token flow

Repo code runs inside the runtime (acceptance commands, `python` in the
session) with the runtime role's credentials and public egress, so that role
must not be able to read the GitHub App private key. Instead:

1. Only the **dispatcher** role may `GetSecretValue` on the App secret
   (`GITHUB_APP_SECRET_ARN` is set on the dispatcher, not the runtime).
2. The dispatcher mints a short-lived installation token scoped to the single
   repo in the event (`github_app.get_installation_token(repository=...)`) and
   passes it to the runtime as `github_token` in the invoke payload.
3. The runtime uses that token only; the orchestrator and Claude's Bash also
   strip AWS credentials from the environment of repo code.

`test_terraform_policy.py` (in `lambda/`) asserts `GetSecretValue` appears only
in the dispatcher policy.

### Dispatcher packaging

`data.archive_file.dispatcher_stub` zips `lambda/dispatcher.py` together with
`../agent/github_app.py`. `github_app.py` imports `boto3` (in the Lambda
runtime), plus `PyJWT` (with `cryptography`) and `requests`, which are **not**
in the Lambda runtime. The dependency layer is built automatically by
`deploy.sh` from `lambda/requirements.txt` (into `build/dispatcher_layer/python`)
and published as `aws_lambda_layer_version.dispatcher_deps`, which the
dispatcher function attaches. If you run Terraform directly without
`deploy.sh`, planning fails with a message to run `deploy.sh` first.

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

## One-shot deploy (`deploy.sh`)

The full 3-stage go-live sequence — create the ECR repo if it doesn't exist,
build & push the agent image via CodeBuild, then `terraform apply` pinned to the
just-pushed image digest — can be run with a single command:

```bash
cd .github/agentcore/terraform
../deploy.sh <aws-account-id> [region]   # region defaults to us-east-1
```

The only runtime inputs are the account id and (optionally) the region;
everything else — image URI, digest pin, resource names — is derived.

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
