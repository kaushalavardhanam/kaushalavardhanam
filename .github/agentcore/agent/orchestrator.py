"""Deterministic orchestrator: requirements -> plan -> execute -> verify.

Python owns the control flow; Claude (via :class:`claude_runner.ClaudeRunner`)
does the thinking and editing inside fresh, guarded sessions:

  1. REQUIREMENTS  read-only session -> numbered requirements, verify commands,
                   scope constraints (schema-validated structured output)
  2. BASELINE      the orchestrator runs the verify commands BEFORE any change,
                   so pre-existing failures are not blamed on the agent
  3. PLAN          read-only session explores the repo -> sub-tasks; the plan is
                   checked by ``plan.validate_plan`` and re-planned with the
                   violation list until it passes (bounded)
  4. EXECUTE       per sub-task: a session that may edit only that sub-task's
                   files; the orchestrator reverts anything else it touched,
                   runs the acceptance command ITSELF, and commits on pass.
                   Two failed attempts -> the sub-task is re-planned into
                   smaller sub-tasks (bounded depth); dependents of a failed
                   sub-task are skipped.
  5. VERIFY        re-run the verify commands; compare with the baseline.

Commits stay local; the caller (``jobs.py``) decides whether and where to push.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import plan as planlib
from claude_runner import ClaudeRunner, RunResult
from plan import Command, Requirements, SubTask

logger = logging.getLogger(__name__)

MAX_PLAN_ATTEMPTS = 3
MAX_EXEC_ATTEMPTS = 2
MAX_RESPLIT_DEPTH = 2
COMMAND_TIMEOUT_SECONDS = 900
OUTPUT_TAIL_CHARS = 4000

DONE, FAILED, SKIPPED = "done", "failed", "skipped"

SYSTEM_APPEND = (
    "You are running headlessly inside an automated pipeline that turns GitHub "
    "issues and reviewer feedback into pull requests. There is no human to ask: "
    "make reasonable decisions from the code and state them. Treat the "
    "repository source as ground truth over any description of it. Do not "
    "commit, push, or install packages — the pipeline does that."
)


class OrchestrationError(RuntimeError):
    """A step that cannot proceed (no requirements, no acceptable plan)."""


@dataclass
class Ask:
    """What to build: an issue, or reviewer feedback on an existing PR."""

    kind: str  # "issue" | "fix"
    repo: str
    title: str
    body: str
    feedback: str = ""
    feedback_author: str = ""
    review_comments: List[str] = field(default_factory=list)
    changed_files: List[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.kind != "fix":
            return f"GitHub issue: {self.title}\n\n{self.body or '(no body)'}"
        lines = [
            f"An existing pull request implements this GitHub issue:\n{self.title}\n\n{self.body or '(no body)'}",
            "",
            "Files the PR already changes: " + (", ".join(self.changed_files) or "(none)"),
            "",
            f"Reviewer @{self.feedback_author} ran it and reported:",
            self.feedback or "(no details)",
        ]
        if self.review_comments:
            lines += ["", "Inline review comments:", *[f"- {c}" for c in self.review_comments]]
        lines += ["", "The ask is to fix what the reviewer reported, within the issue's scope."]
        return "\n".join(lines)


@dataclass
class CommandResult:
    command: Command
    ok: bool
    output: str


@dataclass
class TaskRecord:
    task: SubTask
    status: str
    detail: str = ""
    depth: int = 0
    commit: str = ""


@dataclass
class Outcome:
    requirements: Requirements
    records: List[TaskRecord]
    baseline: List[CommandResult]
    verify: List[CommandResult]
    commits: List[str]
    cost_usd: float
    notes: List[str] = field(default_factory=list)

    @property
    def regressions(self) -> List[CommandResult]:
        """Verify commands that pass at baseline but fail now."""
        base = {(r.command.cwd, r.command.command): r.ok for r in self.baseline}
        return [r for r in self.verify if not r.ok and base.get((r.command.cwd, r.command.command), False)]

    @property
    def unresolved(self) -> List[CommandResult]:
        return [r for r in self.verify if not r.ok]

    @property
    def complete(self) -> bool:
        top_level_ok = all(r.status == DONE for r in self.records if r.depth == 0)
        return top_level_ok and not self.unresolved


def _is_test_path(path: str) -> bool:
    """True for test files: test_*.py, *_test.py, conftest.py, or anything under tests/."""
    parts = path.replace("\\", "/").split("/")
    name = parts[-1]
    return (name == "conftest.py" or (name.startswith("test_") and name.endswith(".py"))
            or name.endswith("_test.py") or any(p in ("tests", "test") for p in parts[:-1]))


def _tail(text: str) -> str:
    return text if len(text) <= OUTPUT_TAIL_CHARS else "...\n" + text[-OUTPUT_TAIL_CHARS:]


class Orchestrator:
    def __init__(self, clone_dir: str, runner: ClaudeRunner, *, budget_usd: float = 20.0,
                 step_budget_usd: float = 3.0, commit_trailers: Sequence[str] = (),
                 command_env: Optional[Dict[str, str]] = None) -> None:
        self.clone_dir = clone_dir
        self.runner = runner
        self.budget_usd = budget_usd
        self.step_budget_usd = step_budget_usd
        self.trailers = list(commit_trailers)
        self.command_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(command_env or {})}
        self.spent = 0.0
        self.notes: List[str] = []

    # ------------------------------------------------------------------ #
    # Infrastructure
    # ------------------------------------------------------------------ #

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.clone_dir, check=True,
                              capture_output=True, text=True).stdout

    def tree(self) -> set:
        return {p for p in self._git("ls-files").splitlines() if p}

    def run_command(self, cmd: Command) -> CommandResult:
        cwd = os.path.join(self.clone_dir, cmd.cwd)
        try:
            proc = subprocess.run(shlex.split(cmd.command), cwd=cwd, env=self.command_env,
                                  capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS)
            out = (proc.stdout or "") + (proc.stderr or "")
            return CommandResult(cmd, proc.returncode == 0, _tail(out))
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CommandResult(cmd, False, f"{type(exc).__name__}: {exc}")

    def _session(self, prompt: str, **kw) -> RunResult:
        remaining = self.budget_usd - self.spent
        if remaining <= 0.05:
            raise OrchestrationError(f"budget of ${self.budget_usd:.2f} exhausted")
        result = self.runner.run(prompt, system_append=SYSTEM_APPEND,
                                 max_budget_usd=min(self.step_budget_usd, remaining), **kw)
        self.spent += result.cost_usd
        return result

    def _changed_paths(self) -> List[str]:
        out = self._git("status", "--porcelain", "--untracked-files=all", "-z")
        paths, entries = [], out.split("\0")
        i = 0
        while i < len(entries):
            entry = entries[i]
            if len(entry) > 3:
                paths.append(entry[3:])
                if entry[0] in "RC":  # rename/copy: next entry is the source path
                    i += 1
            i += 1
        return paths

    def _revert_out_of_scope(self, allowed: Sequence[str]) -> List[str]:
        """Undo every change outside ``allowed``; returns what was reverted."""
        allowed_set, reverted = set(allowed), []
        tracked = self.tree()
        for path in self._changed_paths():
            if path in allowed_set:
                continue
            reverted.append(path)
            if path in tracked:
                self._git("checkout", "HEAD", "--", path)
            else:
                abs_path = os.path.join(self.clone_dir, path)
                if os.path.isdir(abs_path):
                    shutil.rmtree(abs_path, ignore_errors=True)
                elif os.path.exists(abs_path):
                    os.remove(abs_path)
        return reverted

    def _reset_hard(self) -> None:
        self._git("reset", "--hard", "HEAD")
        self._git("clean", "-fdq")

    def _commit(self, task: SubTask) -> str:
        # Stage ONLY the declared files; anything else the acceptance run left
        # behind (caches, artifacts) is discarded rather than committed.
        tracked = self.tree()
        present = [p for p in task.write_paths
                   if p in tracked or os.path.exists(os.path.join(self.clone_dir, p))]
        if present:
            self._git("add", "-A", "--", *present)
        self._revert_out_of_scope(task.write_paths)
        if not self._git("diff", "--cached", "--name-only"):
            return ""
        message = f"{task.title}\n\n{task.goal}"
        if self.trailers:
            message += "\n\n" + "\n".join(self.trailers)
        self._git("commit", "-q", "-m", message)
        return self._git("rev-parse", "HEAD").strip()

    # ------------------------------------------------------------------ #
    # Model steps
    # ------------------------------------------------------------------ #

    def requirements(self, ask: Ask) -> Requirements:
        prompt = f"""{ask.describe()}

Read the repository as needed, then extract the requirements for this ask.

- requirements: concrete, independently checkable outcomes. Where the ask \
already numbers or lists things (cases to cover, bugs reported, "Done when" \
items), keep one requirement per item, ids R1, R2, ...
- verify: commands that prove the WHOLE ask is done (e.g. the issue's "Done \
when" checks, or the command the reviewer said fails). Each is one command \
starting with pytest, python -m pytest, python -m unittest or python -m \
py_compile, with cwd set to the repo-relative directory to run it from.
- constraints: scope rules the ask states (e.g. "tests only; do not modify \
vocabulary.py")."""
        result = self._session(prompt, write_paths=None, schema=planlib.REQUIREMENTS_SCHEMA,
                               max_turns=30)
        if not result.ok:
            raise OrchestrationError(f"requirements step failed ({result.subtype}): {result.errors}")
        reqs = Requirements.parse(result.structured)
        if not reqs.items:
            raise OrchestrationError("requirements step produced no requirements")
        tree = self.tree()
        kept = []
        for cmd in reqs.verify:
            problem = planlib.command_violation(cmd, tree, allow_compile=True)
            if problem:
                self.notes.append(f"Dropped verify command: {problem}")
            else:
                kept.append(cmd)
        reqs.verify = kept
        return reqs

    def _plan_prompt(self, ask_text: str, reqs: Requirements, focus: str,
                     previous: Optional[dict], violations: Sequence[str]) -> str:
        req_lines = "\n".join(f"- {r.id}: {r.text}" for r in reqs.items)
        retry = ""
        if violations:
            retry = ("\n\nYour previous plan was REJECTED by the validator:\n"
                     + "\n".join(f"- {v}" for v in violations)
                     + "\n\nPrevious plan:\n" + json.dumps(previous, indent=1)[:8000]
                     + "\n\nReturn a corrected plan.")
        return f"""{ask_text}
{focus}
Requirements:
{req_lines}

Constraints: {'; '.join(reqs.constraints) or '(none)'}

Explore the repository (read the modules involved, their tests, conftest and \
fixtures) and then split the work into an ORDERED list of sub-tasks. Rules — \
a plan breaking any of them is rejected automatically:
- Every sub-task changes at least one file and at most \
{planlib.MAX_FILES_PER_SUBTASK}. Do not make separate "inspect", "read" or \
"run the tests" sub-tasks: you explore now, and acceptance commands verify.
- files = existing repo paths it modifies; new_files = paths it creates.
- covers = the requirement ids it satisfies; together the sub-tasks cover \
every requirement.
- depends_on lists only EARLIER sub-task ids. Two sub-tasks that change the \
same file must be ordered by depends_on.
- acceptance = ONE command (pytest / python -m pytest / python -m unittest) \
with its cwd, that passes only when this sub-task is done — e.g. a -k selector \
or a single test file. It is checked fail-before / pass-after: the orchestrator \
also runs it with the sub-task's non-test changes stashed, and it must FAIL \
there, so a sub-task that changes behaviour must include a test that exercises \
the change. py_compile is not allowed.
- goal cites real names, signatures and import paths from the code you read.
- Use as FEW sub-tasks as keep each one small and checkable; one is fine for a \
small ask. At most {planlib.MAX_SUBTASKS}.{retry}"""

    def plan(self, ask: Ask, reqs: Requirements, *, focus: str = "",
             requirement_ids: Optional[Sequence[str]] = None) -> List[SubTask]:
        ids = list(requirement_ids or [r.id for r in reqs.items])
        previous, violations = None, []
        for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
            prompt = self._plan_prompt(ask.describe(), reqs, focus, previous, violations)
            result = self._session(prompt, write_paths=None, schema=planlib.PLAN_SCHEMA,
                                   max_turns=40)
            if not result.ok:
                violations = [f"planning session failed: {result.subtype} {result.errors}"]
                continue
            previous = result.structured
            try:
                subtasks = planlib.parse_plan(previous)
            except (KeyError, TypeError, ValueError) as exc:
                violations = [f"plan did not match the schema: {exc}"]
                continue
            violations = planlib.validate_plan(subtasks, ids, self.tree())
            if not violations:
                logger.info("Plan accepted on attempt %d: %d sub-task(s)", attempt, len(subtasks))
                return subtasks
            logger.info("Plan attempt %d rejected: %s", attempt, violations)
        raise OrchestrationError("no acceptable plan after "
                                 f"{MAX_PLAN_ATTEMPTS} attempts: {violations}")

    def _execute_prompt(self, ask: Ask, reqs: Requirements, task: SubTask,
                        done: Sequence[TaskRecord], failure: str) -> str:
        covers = "\n".join(f"- {r.id}: {r.text}" for r in reqs.items if r.id in task.covers)
        prior = "\n".join(f"- {r.task.title}: {r.detail}" for r in done if r.status == DONE) or "(none)"
        retry = (f"\n\nA previous attempt FAILED its acceptance check. Output:\n{failure}\n"
                 "Fix the cause.") if failure else ""
        return f"""{ask.describe()}

You are implementing ONE sub-task of this ask.

Sub-task {task.id}: {task.title}
{task.goal}

Requirements it must satisfy:
{covers}

Constraints: {'; '.join(reqs.constraints) or '(none)'}

You may create or modify ONLY these files: {', '.join(task.write_paths)}
Already completed sub-tasks:
{prior}

Read whatever code you need first. Do not weaken or skip the test; the \
acceptance test must FAIL without your code change and pass with it. When \
done, run the acceptance command yourself and make it pass:
  (cd {task.acceptance.cwd} && {task.acceptance.command})
Finish with a one or two sentence summary of what you changed.{retry}"""

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def _exercises_change(self, task: SubTask) -> bool:
        """Fail-before check: acceptance must FAIL with the non-test changes stashed.

        Test-only sub-tasks have nothing to stash and skip the check.
        """
        code = [p for p in self._changed_paths() if not _is_test_path(p)]
        if not code:
            return True
        before = self._git("stash", "list")
        self._git("stash", "push", "-u", "-q", "--", *code)
        stashed = self._git("stash", "list") != before
        try:
            return not self.run_command(task.acceptance).ok
        finally:
            if stashed:
                self._git("stash", "pop", "-q")

    def _execute(self, ask: Ask, reqs: Requirements, task: SubTask, depth: int,
                 records: List[TaskRecord]) -> bool:
        failure = ""
        for attempt in range(1, MAX_EXEC_ATTEMPTS + 1):
            logger.info("Sub-task %s (depth %d) attempt %d: %s", task.id, depth, attempt, task.title)
            result = self._session(self._execute_prompt(ask, reqs, task, records, failure),
                                   write_paths=task.write_paths, max_turns=60)
            reverted = self._revert_out_of_scope(task.write_paths)
            if reverted:
                self.notes.append(f"Sub-task {task.id}: reverted out-of-scope changes to {reverted}")
            check = self.run_command(task.acceptance)
            if check.ok and not self._exercises_change(task):
                check = CommandResult(task.acceptance, False,
                                      "The acceptance command passes even without your code change: "
                                      "test does not exercise the change. Write a test that FAILS "
                                      "without the change and passes with it.")
            if check.ok:
                sha = self._commit(task)
                records.append(TaskRecord(task, DONE, (result.text or "").strip()[:500], depth, sha))
                return True
            failure = check.output
            logger.info("Sub-task %s acceptance failed (attempt %d)", task.id, attempt)

        self._reset_hard()
        if depth < MAX_RESPLIT_DEPTH:
            try:
                children = self.plan(
                    ask, reqs, requirement_ids=task.covers,
                    focus=(f"\nA previous sub-task was too big to get passing:\n"
                           f"  {task.title}: {task.goal}\n  acceptance: {task.acceptance.command}\n"
                           f"Last failure:\n{failure}\n\nPlan ONLY this sub-task's work, as smaller "
                           f"sub-tasks covering {task.covers}.\n"),
                )
            except OrchestrationError as exc:
                children = []
                self.notes.append(f"Sub-task {task.id}: re-split failed: {exc}")
            if children:
                parent = TaskRecord(task, FAILED, f"split into {len(children)} smaller sub-tasks", depth)
                records.append(parent)
                ok = self._execute_all(ask, reqs, children, depth + 1, records)
                parent.status = DONE if ok else FAILED
                return ok
        records.append(TaskRecord(task, FAILED, _tail(failure), depth))
        return False

    def _execute_all(self, ask: Ask, reqs: Requirements, tasks: Sequence[SubTask], depth: int,
                     records: List[TaskRecord]) -> bool:
        ok_ids, all_ok = set(), True
        for task in tasks:
            blocked = [d for d in task.depends_on if d not in ok_ids]
            if blocked:
                records.append(TaskRecord(task, SKIPPED, f"depends on unfinished {blocked}", depth))
                all_ok = False
                continue
            try:
                if self._execute(ask, reqs, task, depth, records):
                    ok_ids.add(task.id)
                else:
                    all_ok = False
            except OrchestrationError as exc:  # budget exhausted mid-run
                self._reset_hard()
                records.append(TaskRecord(task, FAILED, str(exc), depth))
                all_ok = False
                for rest in tasks[tasks.index(task) + 1:]:
                    records.append(TaskRecord(rest, SKIPPED, str(exc), depth))
                break
        return all_ok

    def run(self, ask: Ask) -> Outcome:
        start = self._git("rev-parse", "HEAD").strip()
        reqs = self.requirements(ask)
        baseline = [self.run_command(c) for c in reqs.verify]
        subtasks = self.plan(ask, reqs)
        records: List[TaskRecord] = []
        self._execute_all(ask, reqs, subtasks, 0, records)
        verify = [self.run_command(c) for c in reqs.verify]
        commits = [c for c in self._git("rev-list", "--reverse", f"{start}..HEAD").split() if c]
        return Outcome(reqs, records, baseline, verify, commits, round(self.spent, 2), self.notes)
