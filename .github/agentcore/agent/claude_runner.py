"""Run one headless Claude Code session (Claude Agent SDK) inside the runtime.

The orchestrator (``orchestrator.py``) owns the control flow; every model step
— requirements, planning, implementing one sub-task — is one call to
:meth:`ClaudeRunner.run`, i.e. a FRESH Claude session with its own tools,
budget, and guard. Nothing carries over between sessions except what the
orchestrator puts in the prompt, so each step is independently reproducible.

Inference goes through Amazon Bedrock with the runtime's execution role (the
SDK bundles the Claude Code CLI, which reads ``CLAUDE_CODE_USE_BEDROCK``); no
Anthropic key exists anywhere.

Guard (:func:`check_tool`, wired as a ``PreToolUse`` hook):
  * read-only sessions may not Edit/Write at all;
  * implement sessions may Edit/Write ONLY the sub-task's declared files;
  * Bash is limited to an allowlist of read/test commands — no git
    commit/push, no network tools, no command substitution.
The hook is defence in depth, not a sandbox: the orchestrator also reverts any
change outside the declared files after each session, and the GitHub token
never enters a Claude session (it is minted only for the final push).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-5-5"

READ_TOOLS = ["Read", "Glob", "Grep", "Bash"]
WRITE_TOOLS = ["Edit", "Write", "MultiEdit"]
# Never needed for repo work; WebFetch/WebSearch would also be an exfil path.
BLOCKED_TOOLS = ["WebFetch", "WebSearch", "Task", "Agent", "NotebookEdit"]

# First word of each allowed Bash segment.
_BASH_READ = {"ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "sort", "uniq",
              "pwd", "echo", "tree", "diff", "file", "stat", "true"}
_BASH_TEST = {"python", "python3", "pytest"}
_GIT_READ = {"status", "diff", "log", "show", "ls-files", "grep", "rev-parse", "blame"}
_FIND_UNSAFE = {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fls"}
_SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\||\n")
_SUBSTITUTION = re.compile(r"`|\$\(|<\(|>\(")


def check_bash(command: str, *, allow_tests: bool) -> Optional[str]:
    """Return a deny reason for ``command``, or None if it is allowed."""
    if _SUBSTITUTION.search(command):
        return "command substitution is not allowed"
    if re.search(r"(^|[^>&0-9])>{1,2}(?!&)", command):
        return "output redirection is not allowed; edit files with the Edit/Write tools"
    allowed = _BASH_READ | (_BASH_TEST if allow_tests else set())
    for segment in _SEGMENT_SPLIT.split(command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            words = shlex.split(segment)
        except ValueError:
            return f"could not parse {segment!r}"
        while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
            words = words[1:]  # leading VAR=value assignments
        if not words:
            continue
        head = os.path.basename(words[0])
        if head == "cd":
            continue
        if head == "git":
            sub = next((w for w in words[1:] if not w.startswith("-")), "")
            if sub not in _GIT_READ:
                return f"git {sub} is not allowed (the orchestrator commits and pushes)"
            continue
        if head == "find" and _FIND_UNSAFE.intersection(words):
            return "find with -exec/-delete is not allowed"
        if head in ("python", "python3") and allow_tests and len(words) > 2 and words[1] == "-m" \
                and words[2] in ("pip", "venv", "ensurepip"):
            return "installing packages is not allowed; the image has the test dependencies"
        if head not in allowed:
            return f"{head!r} is not on the Bash allowlist"
    return None


def check_tool(tool_name: str, tool_input: Dict[str, Any], *, cwd: str,
               write_paths: Optional[Sequence[str]], allow_tests: bool) -> Optional[str]:
    """Return a deny reason for one tool call, or None to let it proceed.

    ``write_paths`` is None for read-only sessions, else the repo-relative files
    this session may create or modify.
    """
    if tool_name in BLOCKED_TOOLS:
        return f"{tool_name} is disabled"
    if tool_name in WRITE_TOOLS:
        if write_paths is None:
            return "this is a read-only session"
        target = tool_input.get("file_path") or tool_input.get("path") or ""
        rel = os.path.relpath(os.path.realpath(os.path.join(cwd, target)), os.path.realpath(cwd))
        if rel.startswith("..") or rel not in set(write_paths):
            return (f"{rel} is outside this sub-task's files {sorted(write_paths)}; "
                    "say so in your summary instead of editing it")
        return None
    if tool_name == "Bash":
        return check_bash(tool_input.get("command", ""), allow_tests=allow_tests)
    return None


@dataclass
class RunResult:
    ok: bool
    subtype: str
    structured: Any = None
    text: str = ""
    cost_usd: float = 0.0
    turns: int = 0
    errors: List[str] = field(default_factory=list)


def bedrock_env(model_id: Optional[str] = None) -> Dict[str, str]:
    """Environment for the bundled Claude Code CLI to use Bedrock."""
    model = model_id or os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID
    if ".anthropic." not in model and not model.startswith("anthropic."):
        raise ValueError(f"BEDROCK_MODEL_ID={model!r} is not an Anthropic model; "
                         "the Claude coding agent requires one")
    return {
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "AWS_REGION": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "ANTHROPIC_MODEL": model,
        # Same model for background/fast calls, so the IAM grant stays one model.
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "DISABLE_TELEMETRY": "1",
        "DISABLE_AUTOUPDATER": "1",
    }


class ClaudeRunner:
    """Runs fresh Claude Code sessions rooted at one checkout."""

    def __init__(self, cwd: str, *, model_id: Optional[str] = None,
                 extra_env: Optional[Dict[str, str]] = None) -> None:
        self.cwd = cwd
        self.env = {
            **bedrock_env(model_id),
            **(extra_env or {}),
            # The CLI needs AWS credentials for Bedrock, but the Bash/python it spawns runs
            # repo code and must not inherit them. Set last so extra_env cannot unset it.
            "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1",
        }

    def run(self, prompt: str, *, system_append: str, write_paths: Optional[Sequence[str]] = None,
            allow_tests: bool = True, schema: Optional[dict] = None,
            max_turns: int = 40, max_budget_usd: float = 2.0) -> RunResult:
        return asyncio.run(self._run(prompt, system_append, write_paths, allow_tests, schema,
                                     max_turns, max_budget_usd))

    async def _run(self, prompt, system_append, write_paths, allow_tests, schema,
                   max_turns, max_budget_usd) -> RunResult:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, ResultMessage, query

        cwd = self.cwd

        async def guard(input_data, tool_use_id, context):  # noqa: ARG001 - SDK signature
            reason = check_tool(input_data.get("tool_name", ""), input_data.get("tool_input") or {},
                                cwd=cwd, write_paths=write_paths, allow_tests=allow_tests)
            if reason is None:
                return {}
            logger.info("Denied %s: %s", input_data.get("tool_name"), reason)
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                           "permissionDecision": "deny",
                                           "permissionDecisionReason": reason}}

        tools = READ_TOOLS + (WRITE_TOOLS if write_paths is not None else [])
        options = ClaudeAgentOptions(
            cwd=cwd,
            env=self.env,
            system_prompt={"type": "preset", "preset": "claude_code", "append": system_append},
            allowed_tools=tools,
            disallowed_tools=BLOCKED_TOOLS + ([] if write_paths is not None else WRITE_TOOLS),
            # Anything not allowed above is refused rather than prompted for.
            permission_mode="dontAsk",
            hooks={"PreToolUse": [HookMatcher(hooks=[guard])]},
            # Repo CLAUDE.md / .claude settings only; never a user's settings.
            setting_sources=["project"],
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            output_format={"type": "json_schema", "schema": schema} if schema else None,
            stderr=lambda line: logger.debug("claude: %s", line),
        )

        final = None
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                final = message
        if final is None:
            return RunResult(ok=False, subtype="no_result", errors=["session ended without a result"])
        result = RunResult(
            ok=not final.is_error and final.subtype == "success",
            subtype=final.subtype,
            structured=final.structured_output,
            text=final.result or "",
            cost_usd=final.total_cost_usd or 0.0,
            turns=final.num_turns,
            errors=list(final.errors or []),
        )
        if schema and result.ok and result.structured is None:
            result.ok, result.subtype = False, "missing_structured_output"
        logger.info("Claude session %s: %s, %d turn(s), $%.2f",
                    final.session_id, result.subtype, result.turns, result.cost_usd)
        return result
