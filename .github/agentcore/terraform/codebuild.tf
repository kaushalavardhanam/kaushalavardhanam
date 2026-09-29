##############################################################################
# CodeBuild image build — build the AgentCore agent container and push to ECR
#
# The AgentCore Runtime (main.tf) needs a linux/arm64 image at
# var.agent_container_image_uri, but that image cannot be built on the local
# machine (no local Docker in this flow). This provisions an AWS-native build:
#
#   terraform apply                         # create the CodeBuild project
#   aws codebuild start-build \             # build + push :latest (and :<sha>)
#     --project-name <name_prefix>-image-build
#   terraform apply -var 'agent_container_image_uri=...:latest'  # real image
#
# CodeBuild runs an ARM64 environment, so the arm64 Dockerfile builds NATIVELY
# — no buildx / QEMU cross-build is needed. privileged_mode is required so the
# build container can run `docker build`.
#
# Source: the PUBLIC GitHub repo is cloned over HTTPS with NO credentials /
# CodeStar connection — public repos clone anonymously. The buildspec lives in
# the repo at .github/agentcore/codebuild/buildspec.yml.
##############################################################################

locals {
  # The ECR repository the image is pushed to. It is created OUTSIDE this
  # Terraform (the repo already exists), so it is referenced by name to build
  # the resource ARN for the push policy rather than managed here.
  ecr_repository_name = "${var.name_prefix}"
  ecr_repository_arn  = "arn:${local.partition}:ecr:${var.aws_region}:${local.account_id}:repository/${local.ecr_repository_name}"
  ecr_registry        = "${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
  ecr_image_uri       = "${local.ecr_registry}/${local.ecr_repository_name}:${var.image_tag}"

  codebuild_project_name = "${var.name_prefix}-image-build"
}

##############################################################################
# CodeBuild service role: ECR auth + push to the agentcore-decomposer repo,
# plus CloudWatch Logs for the build.
##############################################################################

resource "aws_iam_role" "codebuild_image" {
  name = "${var.name_prefix}-image-build"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "codebuild.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = var.tags
}

# ECR: GetAuthorizationToken is registry-wide (no resource scoping possible);
# the push/pull layer actions are scoped to the target repository ARN.
resource "aws_iam_role_policy" "codebuild_image_ecr" {
  name = "EcrPush"
  role = aws_iam_role.codebuild_image.id

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
        Sid    = "EcrPushPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = local.ecr_repository_arn
      }
    ]
  })
}

# CloudWatch Logs for the build's log group/stream.
resource "aws_iam_role_policy" "codebuild_image_logs" {
  name = "CloudWatchLogs"
  role = aws_iam_role.codebuild_image.id

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
        Resource = "arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:/aws/codebuild/${local.codebuild_project_name}:*"
      }
    ]
  })
}

##############################################################################
# CodeBuild project: ARM64 env, privileged for docker build, GitHub source.
##############################################################################

resource "aws_codebuild_project" "image_build" {
  name          = local.codebuild_project_name
  description   = "Builds the AgentCore decomposer agent arm64 container image and pushes it to ECR (no local docker)."
  service_role  = aws_iam_role.codebuild_image.arn
  build_timeout = 30 # minutes

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    # ARM_CONTAINER + an aarch64 standard image builds the arm64 Dockerfile
    # natively. privileged_mode is REQUIRED to run the Docker daemon for
    # `docker build`.
    type                        = "ARM_CONTAINER"
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                       = "aws/codebuild/amazonlinux2-aarch64-standard:3.0"
    image_pull_credentials_type = "CODEBUILD"
    privileged_mode             = true

    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = local.account_id
    }
    environment_variable {
      name  = "AWS_DEFAULT_REGION"
      value = var.aws_region
    }
    environment_variable {
      name  = "ECR_REPOSITORY_NAME"
      value = local.ecr_repository_name
    }
    environment_variable {
      name  = "IMAGE_TAG"
      value = var.image_tag
    }
  }

  source {
    # PUBLIC repo cloned anonymously over HTTPS — no source credential /
    # CodeStar connection is needed for a public GitHub repo.
    type            = "GITHUB"
    location        = "https://github.com/${var.github_repo}.git"
    git_clone_depth = 1
    buildspec       = ".github/agentcore/codebuild/buildspec.yml"

    git_submodules_config {
      fetch_submodules = false
    }
  }

  # Build from the default branch's HEAD by default; override per-build with
  # `aws codebuild start-build --source-version <branch|sha>`.
  source_version = var.image_build_source_version

  logs_config {
    cloudwatch_logs {
      group_name = "/aws/codebuild/${local.codebuild_project_name}"
    }
  }

  tags = var.tags
}

##############################################################################
# Outputs
##############################################################################

output "image_build_project_name" {
  description = "CodeBuild project that builds and pushes the agent image. Run: aws codebuild start-build --project-name <this>."
  value       = aws_codebuild_project.image_build.name
}

output "agent_image_uri" {
  description = "The ECR image URI the build pushes to. Pass this to a re-apply as -var agent_container_image_uri=<this>."
  value       = local.ecr_image_uri
}
