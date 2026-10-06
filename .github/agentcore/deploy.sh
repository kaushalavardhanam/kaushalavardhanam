#!/usr/bin/env bash
#
# One-shot deploy for the AgentCore decomposer backend.
#
# Runtime inputs you provide  (only these two):
#   $1 / AWS_ACCOUNT_ID   - 12-digit account id            (required)
#   $2 / AWS_REGION       - region, default us-east-1       (optional)
#
# Everything else is DERIVED from the Terraform defaults (name_prefix =
# agentcore-decomposer), so it stays in lock-step with the .tf without you
# re-typing names.
#
# Stages:
#   -  build dispatcher Lambda layer   (pip install manylinux x86_64 wheels; no docker)
#   0  ensure ECR repo exists          (create-if-not-exists, idempotent)
#   1  terraform apply  (-target)      -> create the CodeBuild project only
#   2  aws codebuild start-build       -> build+push the arm64 image IN AWS, wait
#   3  terraform apply                 -> re-apply reading the pushed image DIGEST
#                                         from ECR (the source of truth)
#
# Prereqs on the machine you run it from: awscli v2, terraform, pip, AWS creds with
# rights to create ECR/IAM/Lambda/CodeBuild/Bedrock-AgentCore. NO local docker.
# Run from the terraform dir:  cd .github/agentcore/terraform && /path/deploy-agentcore.sh <account-id> [region]

set -euo pipefail

# ---- runtime inputs (the only things you supply) ---------------------------
AWS_ACCOUNT_ID="${1:-${AWS_ACCOUNT_ID:-}}"
AWS_REGION="${2:-${AWS_REGION:-us-east-1}}"

if ! [[ "$AWS_ACCOUNT_ID" =~ ^[0-9]{12}$ ]]; then
  echo "Usage: $0 <12-digit-account-id> [region]   (or set AWS_ACCOUNT_ID / AWS_REGION)" >&2
  exit 2
fi

# ---- everything below is derived, matching the Terraform defaults ----------
NAME_PREFIX="agentcore-decomposer"          # = var.name_prefix default
ECR_REPO="$NAME_PREFIX"                      # codebuild.tf: ecr_repository_name = name_prefix
CODEBUILD_PROJECT="${NAME_PREFIX}-image-build"
IMAGE_TAG="latest"                           # = var.image_tag default
ECR_HOST="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

export AWS_REGION AWS_DEFAULT_REGION="$AWS_REGION"
export TF_VAR_aws_account_id="$AWS_ACCOUNT_ID"
export TF_VAR_aws_region="$AWS_REGION"

echo "== Stage 0: ensure ECR repository '${ECR_REPO}' exists =="
if aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "   exists — reusing"
else
  echo "   not found — creating"
  aws ecr create-repository \
    --repository-name "$ECR_REPO" \
    --region "$AWS_REGION" \
    --image-tag-mutability MUTABLE >/dev/null
  echo "   created"
fi

echo "== Build dispatcher Lambda layer (pip, no docker) =="
rm -rf build/dispatcher_layer
pip install -r lambda/requirements.txt \
  --target build/dispatcher_layer/python \
  --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 \
  --only-binary=:all: --python-version 3.12

echo "== Stage 1: terraform apply (create the CodeBuild project only) =="
terraform init -input=false >/dev/null
# -target the CodeBuild project + its role so the placeholder image URI on the
# AgentCore runtime does not fail this first apply.
terraform apply -input=false -auto-approve \
  -target=aws_codebuild_project.image_build \
  -target=aws_iam_role.codebuild_image \
  -target=aws_iam_role_policy.codebuild_image_ecr \
  -target=aws_iam_role_policy.codebuild_image_logs

echo "== Stage 2: CodeBuild builds + pushes the image (no local docker) =="
BUILD_ID=$(aws codebuild start-build --project-name "$CODEBUILD_PROJECT" \
  --query 'build.id' --output text)
echo "   started: $BUILD_ID  (polling every 15s)"
while :; do
  STATUS=$(aws codebuild batch-get-builds --ids "$BUILD_ID" \
             --query 'builds[0].buildStatus' --output text)
  [ "$STATUS" = "IN_PROGRESS" ] || break
  sleep 15
done
echo "   finished: $STATUS"
if [ "$STATUS" != "SUCCEEDED" ]; then
  echo "!! CodeBuild did not succeed ($STATUS). Logs:" >&2
  echo "   aws codebuild batch-get-builds --ids $BUILD_ID --query 'builds[0].logs.deepLink' --output text" >&2
  exit 1
fi

echo "== Stage 3: read the pushed image DIGEST from ECR, terraform apply =="
DIGEST=$(aws ecr describe-images --repository-name "$ECR_REPO" --region "$AWS_REGION" \
  --image-ids imageTag="$IMAGE_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)
if [ -z "$DIGEST" ] || [ "$DIGEST" = "None" ]; then
  echo "!! could not resolve pushed image digest for ${ECR_REPO}:${IMAGE_TAG}" >&2
  exit 1
fi
IMAGE_URI="${ECR_HOST}/${ECR_REPO}@${DIGEST}"    # pin to immutable digest
echo "   image: $IMAGE_URI"
terraform apply -input=false -auto-approve \
  -var "agent_container_image_uri=${IMAGE_URI}"

echo
echo "== Done. Terraform outputs: =="
terraform output

cat <<EOF

Next (one-time, out of band — not automatable from here):
  1. Grant Bedrock model access for global.anthropic.claude-opus-5 in ${AWS_REGION}.
  2. Populate the GitHub App secret 'agentcore-decomposer/github-app' with
     {app_id, installation_id, private_key}.
Then create an agent-* issue on project 1 and move it to In Progress.
EOF
