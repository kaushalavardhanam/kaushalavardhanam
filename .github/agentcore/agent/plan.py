"""Requirements + plan schemas and the deterministic plan validator.

The planner is a model; whether its plan is acceptable is decided HERE, in
code, so the decomposition guarantees do not depend on the model following
instructions:

* every requirement extracted from the ask is covered by >= 1 sub-task;
* every sub-task changes >= 1 file (no "inspect" / "run the tests" steps —
  exploring happens while planning, verifying happens after executing);
* a sub-task touches at most :data:`MAX_FILES_PER_SUBTASK` files, never CI
  config, and only existing files unless it declares them new;
* every sub-task has an acceptance command the orchestrator can run itself;
* dependencies point only at EARLIER sub-tasks (so list order is a valid
  execution order and there are no cycles);
* two sub-tasks may share a file only if one depends on the other.

A plan with violations goes back to the planner with the violation list.
A small ask legitimately yields ONE sub-task: the rule is "each step is a
verifiable change", not "many steps".
"""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

MAX_SUBTASKS = 12
MAX_FILES_PER_SUBTASK = 4
FORBIDDEN_PREFIXES = (".github/workflows/",)

REQUIREMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        },
        "verify": {
            "type": "array",
            "description": "Commands that prove the whole ask is done (e.g. the issue's 'Done when').",
            "items": {"$ref": "#/$defs/command"},
        },
        "constraints": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["requirements", "verify", "constraints"],
    "additionalProperties": False,
    "$defs": {
        "command": {
            "type": "object",
            "properties": {
                "cwd": {"type": "string", "description": "Repo-relative directory, '.' for the root."},
                "command": {"type": "string"},
            },
            "required": ["cwd", "command"],
            "additionalProperties": False,
        }
    },
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "goal": {"type": "string", "description": "What to change and why, concretely, citing real names from the code."},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "new_files": {"type": "array", "items": {"type": "string"}},
                    "depends_on": {"type": "array", "items": {"type": "integer"}},
                    "covers": {"type": "array", "items": {"type": "string"}},
                    "acceptance": {"$ref": "#/$defs/command"},
                },
                "required": ["id", "title", "goal", "files", "new_files", "depends_on", "covers", "acceptance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subtasks"],
    "additionalProperties": False,
    "$defs": REQUIREMENTS_SCHEMA["$defs"],
}

# Acceptance/verify commands the orchestrator is willing to run itself.
_ALLOWED_COMMANDS = (
    ("pytest",),
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    ("python", "-m", "unittest"),
    ("python3", "-m", "unittest"),
    ("python", "-m", "py_compile"),
    ("python3", "-m", "py_compile"),
)


@dataclass
class Command:
    cwd: str
    command: str

    @classmethod
    def parse(cls, raw: dict) -> "Command":
        return cls(cwd=str(raw.get("cwd") or "."), command=str(raw.get("command") or ""))


@dataclass
class Requirement:
    id: str
    text: str


@dataclass
class Requirements:
    items: List[Requirement]
    verify: List[Command]
    constraints: List[str]

    @classmethod
    def parse(cls, raw: dict) -> "Requirements":
        return cls(
            items=[Requirement(str(r["id"]), str(r["text"])) for r in raw.get("requirements", [])],
            verify=[Command.parse(c) for c in raw.get("verify", [])],
            constraints=[str(c) for c in raw.get("constraints", [])],
        )


@dataclass
class SubTask:
    id: int
    title: str
    goal: str
    files: List[str]
    new_files: List[str]
    depends_on: List[int]
    covers: List[str]
    acceptance: Command

    @classmethod
    def parse(cls, raw: dict) -> "SubTask":
        return cls(
            id=int(raw["id"]),
            title=str(raw["title"]),
            goal=str(raw["goal"]),
            files=[_norm(p) for p in raw.get("files", [])],
            new_files=[_norm(p) for p in raw.get("new_files", [])],
            depends_on=[int(d) for d in raw.get("depends_on", [])],
            covers=[str(c) for c in raw.get("covers", [])],
            acceptance=Command.parse(raw.get("acceptance") or {}),
        )

    @property
    def write_paths(self) -> List[str]:
        return sorted(set(self.files) | set(self.new_files))


def parse_plan(raw: dict) -> List[SubTask]:
    return [SubTask.parse(s) for s in (raw or {}).get("subtasks", [])]


def _norm(path: str) -> str:
    return posixpath.normpath(str(path).strip().lstrip("/"))


def command_violation(cmd: Command, tree: Set[str]) -> str:
    """Why ``cmd`` cannot be run by the orchestrator, or '' if it can."""
    try:
        words = shlex.split(cmd.command)
    except ValueError:
        return f"unparseable command {cmd.command!r}"
    if any(w in ("&&", "||", ";", "|", ">", ">>") for w in words) or "$(" in cmd.command:
        return f"command {cmd.command!r} must be a single command (put the directory in cwd)"
    if not any(tuple(words[: len(p)]) == p for p in _ALLOWED_COMMANDS):
        return (f"command {cmd.command!r} must start with one of: "
                + ", ".join(" ".join(p) for p in _ALLOWED_COMMANDS))
    cwd = _norm(cmd.cwd)
    if cwd.startswith("..") or (cwd != "." and not any(p.startswith(cwd + "/") for p in tree)):
        return f"cwd {cmd.cwd!r} is not a directory in the repo"
    return ""


def validate_plan(subtasks: Sequence[SubTask], requirement_ids: Sequence[str],
                  tree: Set[str]) -> List[str]:
    """Return every rule the plan breaks (empty list == acceptable)."""
    v: List[str] = []
    if not subtasks:
        return ["the plan has no sub-tasks"]
    if len(subtasks) > MAX_SUBTASKS:
        v.append(f"{len(subtasks)} sub-tasks exceeds the maximum of {MAX_SUBTASKS}; merge related steps")

    ids = [s.id for s in subtasks]
    if len(set(ids)) != len(ids):
        v.append("sub-task ids must be unique")
    known_reqs = set(requirement_ids)
    position = {s.id: i for i, s in enumerate(subtasks)}
    owners: Dict[str, List[SubTask]] = {}

    for i, s in enumerate(subtasks):
        tag = f"sub-task {s.id} ({s.title!r})"
        paths = s.write_paths
        if not paths:
            v.append(f"{tag} changes no files; fold inspection into planning and verification "
                     "into acceptance commands instead of separate steps")
        if len(paths) > MAX_FILES_PER_SUBTASK:
            v.append(f"{tag} touches {len(paths)} files (max {MAX_FILES_PER_SUBTASK}); split it")
        for p in paths:
            if p.startswith("..") or p.startswith(FORBIDDEN_PREFIXES):
                v.append(f"{tag} may not change {p}")
        for p in s.files:
            if p not in tree:
                v.append(f"{tag} lists {p} as an existing file but it is not in the repo; "
                         "put new files in new_files")
        for p in s.new_files:
            if p in tree:
                v.append(f"{tag} lists {p} in new_files but it already exists; move it to files")
        unknown = [c for c in s.covers if c not in known_reqs]
        if unknown:
            v.append(f"{tag} covers unknown requirement ids {unknown}")
        if not s.covers:
            v.append(f"{tag} covers no requirement")
        for d in s.depends_on:
            if d not in position or position[d] >= i:
                v.append(f"{tag} depends on {d}, which is not an EARLIER sub-task")
        problem = command_violation(s.acceptance, tree | set(s.new_files))
        if problem:
            v.append(f"{tag} acceptance: {problem}")
        for p in paths:
            owners.setdefault(p, []).append(s)

    covered = {c for s in subtasks for c in s.covers}
    missing = [r for r in requirement_ids if r not in covered]
    if missing:
        v.append(f"requirements {missing} are not covered by any sub-task")

    def depends(a: SubTask, b: SubTask) -> bool:
        seen, stack = set(), list(a.depends_on)
        by_id = {s.id: s for s in subtasks}
        while stack:
            d = stack.pop()
            if d == b.id:
                return True
            if d in seen or d not in by_id:
                continue
            seen.add(d)
            stack.extend(by_id[d].depends_on)
        return False

    for path, group in owners.items():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if not (depends(a, b) or depends(b, a)):
                    v.append(f"sub-tasks {a.id} and {b.id} both change {path} without one "
                             "depending on the other")
    return v
