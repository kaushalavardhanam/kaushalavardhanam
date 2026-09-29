terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      # aws_bedrockagentcore_agent_runtime is a newer resource; it is present
      # in the v6 line of the provider. Pin the floor accordingly.
      source  = "hashicorp/aws"
      version = ">= 6.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.4"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
