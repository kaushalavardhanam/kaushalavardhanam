##############################################################################
# AgentCore decomposer backend — AWS-native IaC
#
# board -> In Progress -> GH Actions kick -> OIDC role -> dispatcher Lambda
#   -> Bedrock AgentCore Runtime agent -> decompose + implement via Bedrock
#   -> open PR
#
# This file provisions the AWS side only. The AgentCore agent container image
# and the real Lambda handler are SEPARATE work items; here they are a
# placeholder image URI (variable) and a minimal inline stub, respectively.
#
# All compute + LLM inference is on AWS. Authentication from GitHub Actions is
# via OIDC (no static AWS keys), mirroring the existing Claude backend.
##############################################################################

data "aws_partition" "current" {}
data "aws_caller_identity" "current" {}

locals {
  account_id = var.aws_account_id
  partition  = data.aws_partition.current.partition

  # The three models the runtime is permitted to invoke, keyed by the global
  # CRIS inference-profile id (the "global." form) the agent invokes with. The
  # foundation-model id is that id with the CRIS routing prefix stripped. The
  # runtime's bedrock_client.py validates BEDROCK_MODEL_ID against exactly this
  # set, so the code allow-list and this IAM grant stay in lockstep.
  bedrock_model_ids = [
    "anthropic.claude-sonnet-5-5",
    "openai.gpt-5.6-sol",
    "openai.gpt-6-astra",
  ]

  # For each model, grant InvokeModel[WithResponseStream] on all four ARN
  # shapes Bedrock's CRIS authorization checks against. Both the region-scoped
  # and the empty-region ("::") foundation-model ARNs are required: CRIS
  # evaluates the request against the empty-account/empty-region variant, which
  # was the exact cause of the prior AccessDeniedException. The inference-profile
  # ARNs (regional + wildcard-region) cover the profile the agent invokes by.
  bedrock_invoke_resources = flatten([
    for fm_id in local.bedrock_model_ids : [
      "arn:${local.partition}:bedrock:::foundation-model/${fm_id}",
      "arn:${local.partition}:bedrock:*::foundation-model/${fm_id}",
      "arn:${local.partition}:bedrock:*:${local.account_id}:inference-profile/global.${fm_id}",
      "arn:${local.partition}:bedrock:${var.aws_region}:${local.account_id}:inference-profile/global.${fm_id}",
    ]
  ])
}

##############################################################################
# 1. Bedrock AgentCore Runtime + execution role
#
# Resource path chosen: the NATIVE aws provider resource
# `aws_bedrockagentcore_agent_runtime` (present in the hashicorp/aws v6 line).
# No awscc_* / null_resource fallback was needed — a first-class typed
# resource exists, which is preferred for correctness.
##############################################################################

resource "aws_iam_role" "agentcore_runtime" {
  name = "${var.name_prefix}-runtime-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock-agentcore.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = local.account_id
          }
        }
      }
    ]
  })

  tags = var.tags
}

# Bedrock model-invoke permissions for the agent's execution role. Grants
# InvokeModel[WithResponseStream] on the FM + inference-profile ARNs (all four
# CRIS-checked shapes) for each of the three allowed models — least-privilege
# to those models, not bedrock:* on all resources.
resource "aws_iam_role_policy" "agentcore_runtime_bedrock" {
  name = "BedrockInvokeModels"
  role = aws_iam_role.agentcore_runtime.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "InvokeAllowedModels"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = local.bedrock_invoke_resources
      }
    ]
  })
}

# CloudWatch Logs for the runtime's execution role.
resource "aws_iam_role_policy" "agentcore_runtime_logs" {
  name = "CloudWatchLogs"
  role = aws_iam_role.agentcore_runtime.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:*"
      }
    ]
  })
}

# Read the GitHub App secret so the agent can open the PR.
resource "aws_iam_role_policy" "agentcore_runtime_secret" {
  name = "ReadGitHubAppSecret"
  role = aws_iam_role.agentcore_runtime.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = aws_secretsmanager_secret.github_app.arn
      }
    ]
  })
}

# ECR pull permissions for the runtime's execution role. AgentCore validates at
# CreateAgentRuntime time that this role can pull the container image from ECR;
# without it CreateAgentRuntime fails with "Access denied while validating ECR
# URI". GetAuthorizationToken is registry-wide (no resource scoping possible);
# the layer/image read actions are scoped to the agentcore-decomposer repo ARN
# (local.ecr_repository_arn, defined alongside the CodeBuild push policy).
resource "aws_iam_role_policy" "agentcore_runtime_ecr" {
  name = "EcrPull"
  role = aws_iam_role.agentcore_runtime.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EcrAuth"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "EcrPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = local.ecr_repository_arn
      }
    ]
  })
}

resource "aws_bedrockagentcore_agent_runtime" "decomposer" {
  agent_runtime_name = replace("${var.name_prefix}_runtime", "-", "_")
  description        = "Decomposes agent-* GitHub issues into sub-tasks and implements them via Bedrock, then opens a PR."
  role_arn           = aws_iam_role.agentcore_runtime.arn

  agent_runtime_artifact {
    container_configuration {
      # Placeholder image; the container build is a separate work item.
      container_uri = var.agent_container_image_uri
    }
  }

  network_configuration {
    # PUBLIC lets the agent reach the GitHub API and Bedrock without a VPC.
    network_mode = "PUBLIC"
  }

  environment_variables = {
    BEDROCK_MODEL_ID      = var.bedrock_model_id
    GITHUB_APP_SECRET_ARN = aws_secretsmanager_secret.github_app.arn
    GITHUB_REPO           = var.github_repo
  }

  tags = var.tags
}

##############################################################################
# 2. Dispatcher Lambda (Python 3.12) + role
#
# Receives {issue_number, title, body, base_branch, repo} and invokes the
# AgentCore Runtime agent. The handler here is a minimal stub — the real
# handler is a later work item.
##############################################################################

data "archive_file" "dispatcher_stub" {
  type        = "zip"
  source_file = "${path.module}/lambda/dispatcher.py"
  output_path = "${path.module}/build/dispatcher.zip"
}

resource "aws_iam_role" "dispatcher_lambda" {
  name = "${var.name_prefix}-dispatcher-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "lambda.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "dispatcher_lambda_logs" {
  name = "CloudWatchLogs"
  role = aws_iam_role.dispatcher_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:*"
      }
    ]
  })
}

# Permission to invoke the AgentCore Runtime agent.
resource "aws_iam_role_policy" "dispatcher_lambda_invoke_agentcore" {
  name = "InvokeAgentCoreRuntime"
  role = aws_iam_role.dispatcher_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock-agentcore:InvokeAgentRuntime"
        ]
        # The runtime ARN plus any version/endpoint qualifier beneath it.
        Resource = [
          aws_bedrockagentcore_agent_runtime.decomposer.agent_runtime_arn,
          "${aws_bedrockagentcore_agent_runtime.decomposer.agent_runtime_arn}/*"
        ]
      }
    ]
  })
}

resource "aws_lambda_function" "dispatcher" {
  function_name    = "${var.name_prefix}-dispatcher"
  description      = "Receives issue payload and invokes the AgentCore decomposer runtime."
  role             = aws_iam_role.dispatcher_lambda.arn
  runtime          = "python3.12"
  handler          = "dispatcher.handler"
  filename         = data.archive_file.dispatcher_stub.output_path
  source_code_hash = data.archive_file.dispatcher_stub.output_base64sha256
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      AGENT_RUNTIME_ARN = aws_bedrockagentcore_agent_runtime.decomposer.agent_runtime_arn
      BEDROCK_MODEL_ID  = var.bedrock_model_id
    }
  }

  tags = var.tags
}

##############################################################################
# 3. OIDC IAM role for GitHub Actions
#
# GitHub Actions federates in via OIDC and assumes this role to invoke the
# dispatcher Lambda. The existing OIDC provider is reused via a data source
# (the Claude backend already created token.actions.githubusercontent.com).
##############################################################################

data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_role" "github_actions_dispatch" {
  name = "${var.name_prefix}-github-oidc"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = data.aws_iam_openid_connect_provider.github.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:*"
          }
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "github_actions_invoke_dispatcher" {
  name = "InvokeDispatcherLambda"
  role = aws_iam_role.github_actions_dispatch.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["lambda:InvokeFunction"]
        Resource = aws_lambda_function.dispatcher.arn
      }
    ]
  })
}

##############################################################################
# 4. Secrets Manager secret for the GitHub App
#
# Holds the GitHub App private key + app/installation IDs the agent uses to
# open the PR. By default the secret is created EMPTY (populate out-of-band).
# No real key is ever placed in code.
##############################################################################

resource "aws_secretsmanager_secret" "github_app" {
  name        = "${var.name_prefix}/github-app"
  description = "GitHub App private key + app_id + installation_id used by the AgentCore agent to open PRs."
  tags        = var.tags
}

resource "aws_secretsmanager_secret_version" "github_app" {
  count         = var.populate_github_app_secret ? 1 : 0
  secret_id     = aws_secretsmanager_secret.github_app.id
  secret_string = var.github_app_secret_value
}

##############################################################################
# Outputs
##############################################################################

output "agent_runtime_arn" {
  description = "ARN of the Bedrock AgentCore Runtime agent."
  value       = aws_bedrockagentcore_agent_runtime.decomposer.agent_runtime_arn
}

output "agent_runtime_id" {
  description = "ID of the Bedrock AgentCore Runtime agent."
  value       = aws_bedrockagentcore_agent_runtime.decomposer.agent_runtime_id
}

output "dispatcher_lambda_arn" {
  description = "ARN of the dispatcher Lambda that GitHub Actions invokes."
  value       = aws_lambda_function.dispatcher.arn
}

output "dispatcher_lambda_name" {
  description = "Name of the dispatcher Lambda."
  value       = aws_lambda_function.dispatcher.function_name
}

output "github_actions_role_arn" {
  description = "ARN of the OIDC role GitHub Actions assumes to invoke the dispatcher."
  value       = aws_iam_role.github_actions_dispatch.arn
}

output "github_app_secret_arn" {
  description = "ARN of the Secrets Manager secret holding the GitHub App credentials."
  value       = aws_secretsmanager_secret.github_app.arn
}
