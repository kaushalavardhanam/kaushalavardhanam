"""Strands Agents decompose-then-implement workflow for the AgentCore agent.

This is a SECOND decomposer implementation, offered ALONGSIDE the LangGraph one
in ``decomposer.py`` — not a replacement. ``agent.py`` selects between them with
the ``DECOMPOSER_IMPL`` env var (default ``langgraph`` preserves current
behaviour); set ``DECOMPOSER_IMPL=strands`` to route here.

Both implement the identical contract so ``agent.py`` and ``git_pr.py`` need no
change: given ``{issue_number, title, body, repo, base_branch}`` this decomposes
the issue into ordered sub-tasks, implements each as a set of whole files, and
returns a final state dict::

    {"subtasks": [...], "files": {path: content, ...},
     "implementation_notes": [...]}

Why a Strands version exists
----------------------------
The LangGraph path had to hand-roll a workaround (issue #13): file CONTENTS
cannot be embedded as JSON string values, because the model stochastically
mis-escapes a quote / backslash / control char inside a large body (making the
envelope unparseable), and a monolithic JSON blob can truncate mid-string. The
sentinel contract (``<<<FILE ...>>>`` / ``<<<END FILE>>>``) is the
defensive fix for that.

Strands removes the failure class STRUCTURALLY instead of defending against it.
Its **native structured output** drives the model through Bedrock tool-use /
constrained decoding: the model fills a schema-validated Pydantic object whose
``content`` field is a proper string the SDK deserialises for us. There is no
JSON envelope we parse by hand and no escaping we can get wrong — a body full of
quotes, newlines and control chars is just a string field value. See
https://strandsagents.com/docs/api/python/strands.models.bedrock/ (structured
output) and the Bedrock provider (``model_id`` is an arbitrary Bedrock /
inference-profile id, e.g. ``global.anthropic.claude-sonnet-4-6``).

Model selection reuses ``bedrock_client``'s resolution + allowlist so both
decomposers invoke exactly the same set of vetted model ids.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

# Reuse the SAME model-id resolution, allowlist, region and token/timeout
# constants as the LangGraph client, so the two decomposers stay in lockstep on
# which models the runtime may invoke and how it reaches Bedrock.
from bedrock_client import (
    ALLOWED_MODEL_IDS,
    DEFAULT_MAX_TOKENS,
    BEDROCK_CONNECT_TIMEOUT_SECONDS,
    BEDROCK_READ_TIMEOUT_SECONDS,
    get_model_id,
    get_region,
)

logger = logging.getLogger(__name__)

MAX_SUBTASKS = 12


# --------------------------------------------------------------------------- #
# Same public state shape as decomposer.py
# --------------------------------------------------------------------------- #


class SubTask(TypedDict):
    """A single ordered unit of work produced by the decomposer."""

    id: int
    title: str
    description: str


class DecomposeState(TypedDict, total=False):
    """Final state returned to ``agent.py`` — identical keys to the LangGraph
    decomposer's ``DecomposeState`` so downstream code is unchanged."""

    issue_number: int
    title: str
    body: str
    repo: str
    base_branch: str

    subtasks: List[SubTask]
    files: Dict[str, str]
    implementation_notes: List[str]
    error: Optional[str]


# --------------------------------------------------------------------------- #
# Structured-output schemas (the whole point: schema-validated file objects)
# --------------------------------------------------------------------------- #


class _SubTaskSchema(BaseModel):
    """One ordered sub-task. Metadata only — never carries file content."""

    id: int = Field(description="1-based ordinal of this sub-task")
    title: str = Field(description="Short imperative title")
    description: str = Field(
        description="What to do and why, concretely, for this sub-task"
    )


class _Decomposition(BaseModel):
    """The ordered list of sub-tasks the issue decomposes into."""

    subtasks: List[_SubTaskSchema] = Field(
        description=(
            f"Ordered list of at most {MAX_SUBTASKS} sub-tasks; earlier "
            f"sub-tasks are prerequisites of later ones"
        )
    )


class _GeneratedFile(BaseModel):
    """One whole file to write.

    ``content`` is an ordinary string field: the SDK deserialises it from the
    model's tool-use output, so quotes / newlines / control characters in the
    body need no escaping and cannot break parsing. This is the structural
    replacement for the LangGraph sentinel contract.
    """

    path: str = Field(
        description="Repo-relative POSIX path (no leading '/', no '..')"
    )
    content: str = Field(
        description="The COMPLETE new content of the file, verbatim (not a diff)"
    )


class _ImplementResult(BaseModel):
    """The set of files produced for one sub-task, plus a one-line note."""

    files: List[_GeneratedFile] = Field(
        description="Every file created or modified in this sub-task"
    )
    note: str = Field(
        default="",
        description="One-line summary of what this sub-task did",
    )


# --------------------------------------------------------------------------- #
# Path safety (mirrors decomposer._sanitize_path)
# --------------------------------------------------------------------------- #


def _sanitize_path(path: str) -> Optional[str]:
    """Return a safe repo-relative POSIX path, or None if it escapes the repo."""
    path = path.strip().lstrip("/")
    if not path:
        return None
    parts = path.split("/")
    if ".." in parts:
        return None
    parts = [p for p in parts if p not in ("", ".")]
    if not parts:
        return None
    return "/".join(parts)


# --------------------------------------------------------------------------- #
# Strands agent factory
# --------------------------------------------------------------------------- #


DECOMPOSE_SYSTEM = (
    "You are a senior software engineer decomposing a GitHub issue into an "
    "ordered list of concrete, self-contained implementation sub-tasks. Each "
    "sub-task should be small enough to implement as a coherent set of file "
    "edits, and ordered so that later sub-tasks may depend on earlier ones."
)

IMPLEMENT_SYSTEM = (
    "You are an expert software engineer implementing one sub-task of a larger "
    "GitHub issue. For every file you create or modify in this step you return "
    "its COMPLETE new content (never a diff, never a partial file), following "
    "the existing conventions of the repository. Return the files as structured "
    "data: each file is an object with a repo-relative 'path' and its full "
    "'content' as a plain string — do not wrap content in JSON, markdown fences "
    "or any escaping."
)


class StrandsDecomposer:
    """Wraps a Strands ``Agent`` bound to the configured Bedrock model.

    The heavy Strands imports happen inside ``__init__`` so the module can be
    imported (and its pure helpers/schemas unit-tested) without the
    ``strands-agents`` package installed. ``agent.py`` only constructs this when
    ``DECOMPOSER_IMPL=strands``.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model_id = model_id or get_model_id()
        if self.model_id not in ALLOWED_MODEL_IDS:
            raise ValueError(
                f"Unsupported BEDROCK_MODEL_ID {self.model_id!r}. "
                f"Allowed model ids are: {', '.join(ALLOWED_MODEL_IDS)}."
            )
        self.region_name = region_name or get_region()
        self.max_tokens = max_tokens

        # Imported lazily: see class docstring.
        from botocore.config import Config
        from strands import Agent
        from strands.models import BedrockModel

        # NOTE: temperature is intentionally NOT set. Bedrock's current Anthropic
        # (Claude Sonnet 5.5) and OpenAI models reject a non-default temperature
        # with a ValidationException; letting the provider default stand avoids
        # it — the same guard bedrock_client.py applies for the hand-rolled path.
        boto_config = Config(
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=BEDROCK_CONNECT_TIMEOUT_SECONDS,
            read_timeout=BEDROCK_READ_TIMEOUT_SECONDS,
        )
        self._model = BedrockModel(
            model_id=self.model_id,
            region_name=self.region_name,
            max_tokens=max_tokens,
            boto_client_config=boto_config,
        )
        self._Agent = Agent
        logger.info(
            "StrandsDecomposer ready (model=%s region=%s)",
            self.model_id,
            self.region_name,
        )

    def _new_agent(self, system_prompt: str):
        return self._Agent(model=self._model, system_prompt=system_prompt)

    def _structured(self, agent, output_model, prompt: str):
        """Call Strands structured output, tolerating both SDK API shapes.

        Newer SDKs: ``agent(prompt, structured_output_model=Model).structured_output``.
        Older SDKs: ``agent.structured_output(Model, prompt)`` returning the model.
        """
        # Preferred: the call-with-kwarg form (returns an AgentResult).
        try:
            result = agent(prompt, structured_output_model=output_model)
        except TypeError:
            result = None
        if result is not None:
            structured = getattr(result, "structured_output", None)
            if structured is not None:
                return structured
            # Some versions return the model instance directly.
            if isinstance(result, output_model):
                return result
        # Fallback: the dedicated method (older SDKs).
        return agent.structured_output(output_model, prompt)

    def decompose(self, state: DecomposeState) -> List[SubTask]:
        logger.info("Decomposing issue #%s (strands)", state.get("issue_number"))
        prompt = f"""Decompose the following GitHub issue into an ORDERED list of \
implementation sub-tasks.

Repository: {state.get('repo')}
Issue #{state.get('issue_number')}: {state.get('title')}

Issue body:
{state.get('body') or '(no body provided)'}

Produce at most {MAX_SUBTASKS} sub-tasks, ordered so earlier ones are \
prerequisites of later ones."""

        agent = self._new_agent(DECOMPOSE_SYSTEM)
        decomposition = self._structured(agent, _Decomposition, prompt)

        subtasks: List[SubTask] = []
        for i, item in enumerate(decomposition.subtasks[:MAX_SUBTASKS], start=1):
            subtasks.append(
                SubTask(
                    id=int(item.id or i),
                    title=str(item.title or f"Sub-task {i}"),
                    description=str(item.description or ""),
                )
            )
        return subtasks

    def implement(
        self, state: DecomposeState, task: SubTask, files: Dict[str, str]
    ) -> "tuple[Dict[str, str], str]":
        """Implement one sub-task; return the file blocks it produced + a note."""
        existing = "\n".join(sorted(files)) or "(none yet)"
        existing_context = ""
        if files:
            existing_context = "\n\n".join(
                f"--- {path} ---\n{content}" for path, content in files.items()
            )[:20000]

        prompt = f"""You are implementing ONE sub-task for issue \
#{state.get('issue_number')} ("{state.get('title')}") in repo \
{state.get('repo')}.

Sub-task {task['id']}: {task['title']}
{task['description']}

Files already generated by earlier sub-tasks:
{existing}

Contents of those files (for consistency; you may return updated versions):
{existing_context or '(none)'}

Return the complete set of files to create or modify for THIS sub-task. Each \
file must carry its repo-relative POSIX path and its FULL content."""

        agent = self._new_agent(IMPLEMENT_SYSTEM)
        result = self._structured(agent, _ImplementResult, prompt)

        produced: Dict[str, str] = {}
        for gf in result.files:
            safe = _sanitize_path(gf.path)
            if safe is None:
                logger.warning("Skipping unsafe path from model: %s", gf.path)
                continue
            produced[safe] = gf.content
        return produced, (result.note or "implemented")


# --------------------------------------------------------------------------- #
# Public entrypoint (same signature as decomposer.run_decomposition)
# --------------------------------------------------------------------------- #


def run_decomposition(
    *,
    issue_number: int,
    title: str,
    body: str,
    repo: str,
    base_branch: str,
    decomposer: Optional["StrandsDecomposer"] = None,
) -> DecomposeState:
    """Run the full decompose->implement workflow (Strands) and return state.

    Signature mirrors ``decomposer.run_decomposition`` except the injected
    dependency is a :class:`StrandsDecomposer` (tests pass a fake). Returns the
    same ``{subtasks, files, implementation_notes}`` shape ``agent.py`` expects.
    """
    dec = decomposer or StrandsDecomposer()

    state: DecomposeState = {
        "issue_number": issue_number,
        "title": title,
        "body": body,
        "repo": repo,
        "base_branch": base_branch,
    }

    subtasks = dec.decompose(state)
    if not subtasks:
        raise RuntimeError("decomposition produced no sub-tasks")
    logger.info("Decomposed into %d sub-task(s)", len(subtasks))

    files: Dict[str, str] = {}
    notes: List[str] = []
    for task in subtasks:
        logger.info("Implementing sub-task %s: %s", task["id"], task["title"])
        produced, note = dec.implement(state, task, files)
        files.update(produced)
        notes.append(f"{task['id']}. {task['title']} — {note}")

    if not files:
        raise RuntimeError("implementation produced no files")

    logger.info("Implementation complete: %d file(s)", len(files))
    return {
        "issue_number": issue_number,
        "title": title,
        "body": body,
        "repo": repo,
        "base_branch": base_branch,
        "subtasks": subtasks,
        "files": files,
        "implementation_notes": notes,
    }
