# Worker git credentials (so agents can open PRs)

Self-hosted Cursor / AgentCore workers **do not** receive Cursor’s GitHub App token and **do not** see the Actions `GITHUB_TOKEN`. Push and `autoCreatePR` use whatever credentials are already on the worker VM. That is Cursor’s self-hosted design.

Widening `.github/workflows/cursor-agent-in-progress.yml` `permissions:` (for example `contents: write`) does **not** let an agent open a PR. That token exists only in the kickoff job, which never has the agent’s commits.

## What to put on the AgentCore worker

Create a **fine-grained PAT** (or GitHub App installation token) for `kaushalavardhanam/kaushalavardhanam`:

| Permission | Access |
|---|---|
| Contents | Read and write |
| Pull requests | Read and write |
| Metadata | Read (automatic) |

Do **not** put this token in git or `config.yaml`.

Set **one** of these on the **worker session / image** (same place `CURSOR_API_KEY` already lives):

```text
CURSOR_GIT_TOKEN=github_pat_...
```

Aliases the helper also accepts: `GH_TOKEN`, `GH_WORKER_TOKEN`.

At worker start (before the agent runs `git push`):

```bash
.github/scripts/configure-worker-git.sh
```

That installs `.github/scripts/git-credential-github-env` as the HTTPS helper for `github.com`.

## Check

On a claimed worker:

```bash
.github/scripts/configure-worker-git.sh
git ls-remote --heads origin
git push -u origin HEAD
```

`git ls-remote` can succeed on a public repo **without** write access. `git push` is the real check.

## What will not work

- Raising the kickoff workflow’s `GITHUB_TOKEN` scopes
- Expanding `GH_PROJECT_TOKEN` **only** as an Actions secret (it is not injected into AgentCore)
- Relying on Cursor hosted-VM App tokens — those are not injected into self-hosted workers

You **can** reuse the same PAT as `GH_PROJECT_TOKEN` **if** you also install it on the worker and add Contents + Pull requests write. Projects: Read stays required for the 5-minute scan.

## Cursor team-pool minting (optional alternative)

If you switch from AgentCore to Cursor’s `agent worker --pool` with `--mint-github-token` / `--clone-git-repos`, a team admin can enable GitHub token minting in the Cloud Agents dashboard. That still does not apply to this AgentCore image until the worker is started that way.
