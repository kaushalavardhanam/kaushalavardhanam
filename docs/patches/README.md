# cursor-cookbook AgentCore git-token patch

Applies on `skopp002/cursor-cookbook` at `18d6d355dbd132b501f0fe0760d2e66161018809`.

Adds worker git-token fetch to `self-hosted-cloud-agent/agentcore` (adapter, Terraform, README). See `.github/WORKER_GIT.md` for the live-lab contract.

This worker cannot push that repository (fine-grained PAT is limited to `kaushalavardhanam/kaushalavardhanam`). Apply and push from an account with write access to the cookbook.
