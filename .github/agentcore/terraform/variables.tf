variable "aws_region" {
  description = "AWS region to deploy the AgentCore decomposer backend into. Bedrock model access and the AgentCore Runtime must be available here."
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "The 12-digit AWS account ID the backend deploys to. Used to build Bedrock inference-profile / foundation-model ARNs and the OIDC role ARN. Not a secret, but parameterized so it is not published in this public repo."
  type        = string

  validation {
    condition     = can(regex("^[0-9]{12}$", var.aws_account_id))
    error_message = "aws_account_id must be a 12-digit AWS account ID."
  }
}

variable "name_prefix" {
  description = "Prefix applied to created resource names, so multiple deployments (e.g. per environment) do not collide."
  type        = string
  default     = "agentcore-decomposer"
}

variable "github_repo" {
  description = "The GitHub repo (owner/name) whose Actions workflows may assume the OIDC role. The trust policy is scoped to repo:<this>:*."
  type        = string
  default     = "kaushalavardhanam/kaushalavardhanam"
}

variable "github_oidc_sub" {
  description = "The exact `token.actions.githubusercontent.com:sub` claim value the OIDC role trust policy matches. GitHub's real OIDC token carries the repo in its numeric @id form (repo:<owner>@<owner_id>/<repo>@<repo_id>:*), NOT the plain repo:<owner>/<repo>:* form; matching on the plain form causes AssumeRoleWithWebIdentity denial. Defaults to this repo's @id subject."
  type        = string
  default     = "repo:kaushalavardhanam@325055035/kaushalavardhanam@956097101:*"
}

variable "bedrock_model_id" {
  description = "Bedrock model / inference-profile (global CRIS) id Claude Code runs on inside the runtime. Must be an Anthropic model covered by local.bedrock_model_ids."
  type        = string
  default     = "global.anthropic.claude-sonnet-5-5"

  validation {
    condition     = contains(["global.anthropic.claude-sonnet-5-5"], var.bedrock_model_id)
    error_message = "bedrock_model_id must be global.anthropic.claude-sonnet-5-5 (extend local.bedrock_model_ids to grant another Anthropic model)."
  }
}

variable "agent_container_image_uri" {
  description = "ECR image URI for the containerized AgentCore Runtime agent. The agent code + image build is a SEPARATE work item; this is a placeholder input. Use a fully-qualified ECR URI, e.g. <account>.dkr.ecr.<region>.amazonaws.com/agentcore-decomposer:latest."
  type        = string
  default     = "PLACEHOLDER.dkr.ecr.us-east-1.amazonaws.com/agentcore-decomposer:latest"
}

variable "populate_github_app_secret" {
  description = "Whether to seed the GitHub App secret with a value from var.github_app_secret_value. Leave false (default) to create an EMPTY secret container and populate it out-of-band (console/CLI). Never commit a real key."
  type        = bool
  default     = false
}

variable "github_app_secret_value" {
  description = "JSON string holding the GitHub App private key + app_id + installation_id. ONLY used when populate_github_app_secret = true. Pass via a tfvars file that is NOT committed, or an environment variable. Do not hardcode."
  type        = string
  default     = ""
  sensitive   = true
}

variable "tags" {
  description = "Tags applied to all resources."
  type        = map(string)
  default = {
    Project = "agentcore-decomposer"
    Backend = "aws-bedrock-agentcore"
  }
}

variable "image_tag" {
  description = "Tag applied to the agent container image the CodeBuild project pushes to ECR (in addition to the commit SHA). Re-apply with agent_container_image_uri=<registry>/<repo>:<this> once the build has run."
  type        = string
  default     = "latest"
}

variable "image_build_source_version" {
  description = "The git ref (branch name or commit SHA) CodeBuild clones and builds the image from. Defaults to the branch carrying the agent source + Dockerfile. Override per build with `aws codebuild start-build --source-version <ref>`."
  type        = string
  default     = "agentcore-backend"
}

variable "agent_budget_usd" {
  description = "Upper bound on Bedrock model spend (USD, as reported by the Claude Agent SDK) for ONE job — one issue or one /agent fix comment. Each Claude session is also capped at $3."
  type        = number
  default     = 20
}
