# Worker git credentials (so agents can open PRs)

Self-hosted Cursor / AgentCore workers **do not** receive Cursor’s GitHub App token and **do not** see the Actions `GITHUB_TOKEN`. Push and `autoCreatePR` use whatever credentials are already on the worker VM. That is Cursor’s self-hosted design.

Widening `.github/workflows/cursor-agent-in-progress.yml` `permissions:` (for example `contents: write`) does **not** let an agent open a PR. That token exists only in the kickoff job, which never has the agent’s commits.

This repository is the application repo. The worker image and Terraform live in [cursor-cookbook `self-hosted-cloud-agent/agentcore`](https://github.com/skopp002/cursor-cookbook/tree/main/self-hosted-cloud-agent/agentcore).

The cookbook adapter/README change is in [`docs/patches/cursor-cookbook-agentcore-git-token.patch`](../docs/patches/cursor-cookbook-agentcore-git-token.patch) (based on `18d6d35`). Apply it on that repo — this worker’s PAT is scoped to `kaushalavardhanam/kaushalavardhanam` and cannot push `skopp002/cursor-cookbook`.

```bash
cd cursor-cookbook
git checkout 18d6d355dbd132b501f0fe0760d2e66161018809
git checkout -b cursor/agentcore-worker-git-token
git am ../kaushalavardhanam/docs/patches/cursor-cookbook-agentcore-git-token.patch
git push -u origin cursor/agentcore-worker-git-token
```

## Required AgentCore configuration

This is **not** automated on an older worker image. A session that was already running when the secret was created still has to load the token by hand. After the cookbook adapter change is published:

1. Secrets Manager secret `cursor-agentcore-worker-git-token` exists (JSON `{"CURSOR_GIT_TOKEN":"github_pat_..."}` or plaintext).
2. Runtime env has `CURSOR_GIT_TOKEN_SECRET_ID=<secret ARN>`.
3. Role `cursor_worker-execution-role` can `secretsmanager:GetSecretValue` on that ARN.
4. A **new** worker image (adapter `resolve_git_token()`) is pushed and a **new** session is started.

Until those four are true, agents in this repo should run `.github/scripts/load-worker-git-token.sh` (fetches the secret and installs the credential helper).

### Fine-grained PAT

Create the token from GitHub → Settings → Developer settings → Fine-grained tokens. Resource owner `kaushalavardhanam`, repository `kaushalavardhanam`.

| Permission | Access |
|---|---|
| Contents | Read and write |
| Pull requests | Read and write |
| Workflows | Read and write (this repo’s agents edit workflow YAML) |
| Metadata | Read (automatic) |

Do **not** put this token in git or `config.yaml`.

### Upload / rotate

```bash
# from cursor-cookbook/self-hosted-cloud-agent/agentcore
export CURSOR_GIT_TOKEN=github_pat_...
# secret already exists in account 146666888814
export CREATE_CURSOR_GIT_TOKEN_SECRET=false
make agentcore-put-git-token-secret
make agentcore-terraform-apply
make agentcore-ecr-build-push
# then start a new AgentCore session — the current one will not pick this up
```

## Fallback on this workspace (current image)

```bash
.github/scripts/load-worker-git-token.sh
git ls-remote --heads origin
git push -u origin HEAD
```

Aliases the helper also accepts: `CURSOR_GIT_TOKEN`, `GH_TOKEN`, `GH_WORKER_TOKEN`.

`git ls-remote` can succeed on a public repo **without** write access. `git push` is the real check.

## What will not work

- Raising the kickoff workflow’s `GITHUB_TOKEN` scopes
- Expanding `GH_PROJECT_TOKEN` **only** as an Actions secret (it is not injected into AgentCore)
- Relying on Cursor hosted-VM App tokens — those are not injected into self-hosted workers

You **can** reuse the same PAT as `GH_PROJECT_TOKEN` **if** you also install it on the worker and add Contents + Pull requests write. Projects: Read stays required for the 5-minute scan.

## Cursor team-pool minting (optional alternative)

If you switch from AgentCore to Cursor’s `agent worker --pool` with `--mint-github-token` / `--clone-git-repos`, a team admin can enable GitHub token minting in the Cloud Agents dashboard. That still does not apply to this AgentCore image until the worker is started that way.
