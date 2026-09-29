"""LangGraph decompose-then-implement workflow for the AgentCore agent.

This builds on the multi-agent LangGraph pattern in the repo's
``coding-agent/graph_workflow.py`` (a supervisor node that routes between
worker nodes via a shared ``WorkflowState`` TypedDict and a conditional-edge
router). Here the workflow is specialised for the issue-to-PR pipeline:

    decompose  ->  implement (loops over sub-tasks)  ->  END

* **decompose**: one Bedrock call turns the GitHub issue into an *ordered* list
  of sub-tasks (a strict JSON array).
* **implement**: for each sub-task in order, one Bedrock call produces the set
  of file writes (path -> full new content) for that step; earlier steps'
  files are fed back as context so later steps build on them. The accumulated
  files are what the caller commits.

The LLM prompts are a solid first version and are intentionally the easiest
part to iterate on; the plumbing (Bedrock Messages calls, strict JSON parsing,
ordered state machine) is the part that must be correct.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from bedrock_client import BedrockClient

logger = logging.getLogger(__name__)

MAX_SUBTASKS = 12


class SubTask(TypedDict):
    """A single ordered unit of work produced by the decomposer."""

    id: int
    title: str
    description: str


class DecomposeState(TypedDict, total=False):
    """Shared state threaded through the LangGraph workflow.

    Mirrors the ``WorkflowState`` approach in ``graph_workflow.py``: every node
    returns a partial dict that LangGraph merges into this state.
    """

    issue_number: int
    title: str
    body: str
    repo: str
    base_branch: str

    subtasks: List[SubTask]
    # Accumulated file contents: repo-relative path -> full file content.
    files: Dict[str, str]
    # Human-readable log of what each sub-task did (used for the PR body).
    implementation_notes: List[str]
    error: Optional[str]


# --------------------------------------------------------------------------- #
# JSON extraction helpers
# --------------------------------------------------------------------------- #

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Return the content of the first ```...``` fence, or the text unchanged."""
    match = _JSON_FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def _extract_json_array(text: str) -> list:
    """Parse a JSON array from a model response, tolerating prose/fences."""
    cleaned = _strip_fences(text)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in model response: {text[:300]}")
    return json.loads(cleaned[start : end + 1])


def _extract_json_object(text: str) -> dict:
    """Parse a JSON object from a model response, tolerating prose/fences."""
    cleaned = _strip_fences(text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model response: {text[:300]}")
    return json.loads(cleaned[start : end + 1])


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

DECOMPOSE_SYSTEM = (
    "You are a senior software engineer decomposing a GitHub issue into an "
    "ordered list of concrete, self-contained implementation sub-tasks. Each "
    "sub-task should be small enough to implement as a coherent set of file "
    "edits, and ordered so that later sub-tasks may depend on earlier ones."
)


def _decompose_node(bedrock: BedrockClient):
    def decompose(state: DecomposeState) -> Dict:
        logger.info("Decomposing issue #%s", state.get("issue_number"))
        prompt = f"""Decompose the following GitHub issue into an ORDERED list of \
implementation sub-tasks.

Repository: {state.get('repo')}
Issue #{state.get('issue_number')}: {state.get('title')}

Issue body:
{state.get('body') or '(no body provided)'}

Return ONLY a JSON array (no prose, no markdown fences) of at most {MAX_SUBTASKS} \
objects, each shaped exactly:
  {{"id": <1-based integer>, "title": "<short imperative title>", "description": "<what to do and why, concretely>"}}

Order the array so earlier sub-tasks are prerequisites of later ones."""

        raw = bedrock.invoke(prompt, system=DECOMPOSE_SYSTEM, temperature=0.2)
        try:
            parsed = _extract_json_array(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse decomposition: %s", exc)
            return {"error": f"decomposition parse failure: {exc}"}

        subtasks: List[SubTask] = []
        for i, item in enumerate(parsed[:MAX_SUBTASKS], start=1):
            if not isinstance(item, dict):
                continue
            subtasks.append(
                SubTask(
                    id=int(item.get("id", i)),
                    title=str(item.get("title", f"Sub-task {i}")),
                    description=str(item.get("description", "")),
                )
            )
        if not subtasks:
            return {"error": "decomposition produced no sub-tasks"}

        logger.info("Decomposed into %d sub-task(s)", len(subtasks))
        return {"subtasks": subtasks, "files": {}, "implementation_notes": []}

    return decompose


IMPLEMENT_SYSTEM = (
    "You are an expert software engineer implementing one sub-task of a larger "
    "issue. You output complete file contents (never diffs or partial files) "
    "for every file you create or modify in this step, following the existing "
    "conventions of the repository."
)


def _implement_node(bedrock: BedrockClient):
    def implement(state: DecomposeState) -> Dict:
        subtasks = state.get("subtasks", [])
        files: Dict[str, str] = dict(state.get("files", {}))
        notes: List[str] = list(state.get("implementation_notes", []))

        for task in subtasks:
            logger.info("Implementing sub-task %s: %s", task["id"], task["title"])

            # Give the model the paths already produced so it can build on them
            # and rewrite a file it touched in an earlier step if needed.
            existing = "\n".join(sorted(files)) or "(none yet)"
            existing_context = ""
            if files:
                # Include contents of already-generated files (bounded) so
                # later sub-tasks are consistent with earlier ones.
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

Return ONLY a JSON object (no prose, no markdown fences) shaped exactly:
  {{"files": {{"<repo-relative-path>": "<FULL file content>"}}, "note": "<one-line summary of what you did>"}}

Rules:
- Keys are repo-relative POSIX paths. Do NOT use absolute paths or '..'.
- Values are the COMPLETE new content of each file, not a diff.
- Only include files this sub-task actually creates or changes."""

            raw = bedrock.invoke(prompt, system=IMPLEMENT_SYSTEM, temperature=0.2)
            try:
                result = _extract_json_object(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.error("Sub-task %s parse failure: %s", task["id"], exc)
                return {
                    "error": f"sub-task {task['id']} parse failure: {exc}",
                    "files": files,
                    "implementation_notes": notes,
                }

            step_files = result.get("files", {})
            if isinstance(step_files, dict):
                for path, content in step_files.items():
                    safe = _sanitize_path(str(path))
                    if safe is None:
                        logger.warning("Skipping unsafe path from model: %s", path)
                        continue
                    files[safe] = content if isinstance(content, str) else str(content)

            notes.append(
                f"{task['id']}. {task['title']} — {result.get('note', 'implemented')}"
            )

        if not files:
            return {
                "error": "implementation produced no files",
                "files": files,
                "implementation_notes": notes,
            }

        logger.info("Implementation complete: %d file(s)", len(files))
        return {"files": files, "implementation_notes": notes}

    return implement


def _sanitize_path(path: str) -> Optional[str]:
    """Return a safe repo-relative POSIX path, or None if it escapes the repo."""
    path = path.strip().lstrip("/")
    if not path:
        return None
    parts = path.split("/")
    if any(p in ("", "..", ".") for p in parts):
        # Reject traversal and empty segments.
        if ".." in parts:
            return None
        parts = [p for p in parts if p not in ("", ".")]
        if not parts:
            return None
    return "/".join(parts)


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #


def build_workflow(bedrock: Optional[BedrockClient] = None):
    """Compile the decompose -> implement LangGraph workflow.

    Args:
        bedrock: Optional injected client (tests pass a fake). Defaults to a
            real ``BedrockClient``.

    Returns:
        A compiled LangGraph app with ``.invoke(state)``.
    """
    bedrock = bedrock or BedrockClient()

    graph = StateGraph(DecomposeState)
    graph.add_node("decompose", _decompose_node(bedrock))
    graph.add_node("implement", _implement_node(bedrock))

    graph.set_entry_point("decompose")

    # If decomposition errored, skip implementation and end.
    def after_decompose(state: DecomposeState) -> str:
        return "end" if state.get("error") else "implement"

    graph.add_conditional_edges(
        "decompose", after_decompose, {"implement": "implement", "end": END}
    )
    graph.add_edge("implement", END)

    return graph.compile()


def run_decomposition(
    *,
    issue_number: int,
    title: str,
    body: str,
    repo: str,
    base_branch: str,
    bedrock: Optional[BedrockClient] = None,
) -> DecomposeState:
    """Run the full decompose->implement workflow and return the final state."""
    app = build_workflow(bedrock)
    initial: DecomposeState = {
        "issue_number": issue_number,
        "title": title,
        "body": body,
        "repo": repo,
        "base_branch": base_branch,
    }
    result = app.invoke(initial)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result  # type: ignore[return-value]
