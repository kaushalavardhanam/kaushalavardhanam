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

# Literal control characters (U+0000–U+001F) are legal NOWHERE in strict JSON
# but routinely appear INSIDE model-generated string values — e.g. a file's full
# content returned as {"files": {"x.md": "line1\nline2"}} with a real newline
# instead of an escaped \n. json.loads(strict=True) rejects them ("Invalid
# control character", "Unterminated string"); _loads_tolerant below recovers by
# permitting them (strict=False) and, as a last resort, re-escaping the control
# chars that fall inside a JSON string span.


def _strip_fences(text: str) -> str:
    """Return the content of the first ```...``` fence, or the text unchanged."""
    match = _JSON_FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def _escape_control_chars_in_strings(payload: str) -> str:
    """Escape literal control characters that appear inside JSON string values.

    Walks the text tracking whether we are inside a (double-quoted) JSON string,
    honouring backslash escapes, and replaces any raw control character found
    inside a string with its JSON escape sequence (``\\n``, ``\\t``, or the
    ``\\uXXXX`` form). Control characters outside strings (structural whitespace)
    are left untouched. This is the last-resort repair for model output whose
    string values contain unescaped newlines/tabs/control bytes.
    """
    out: List[str] = []
    in_string = False
    escaped = False
    for ch in payload:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ord(ch) < 0x20:
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\t":
                    out.append("\\t")
                elif ch == "\r":
                    out.append("\\r")
                else:
                    out.append("\\u%04x" % ord(ch))
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def _loads_tolerant(payload: str):
    """Parse a JSON document, tolerating control characters in string values.

    Tiered so well-formed responses are unaffected:
      1. strict parse (fast path, unchanged behaviour for valid JSON);
      2. ``strict=False`` — permits raw control chars inside strings, which is
         the common Bedrock/Sonnet case (file content with real newlines);
      3. re-escape control chars found inside string spans, then parse.
    Raises the *last* ``JSONDecodeError`` (from the tolerant attempt) if all
    tiers fail, so the caller surfaces the most informative error.
    """
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass
    try:
        # strict=False allows literal \t and \n (and other control chars) to
        # appear unescaped within JSON strings.
        return json.loads(payload, strict=False)
    except json.JSONDecodeError:
        pass
    # Last resort: explicitly re-escape control chars inside strings. This also
    # rescues chars strict=False still rejects when they break token scanning.
    return json.loads(_escape_control_chars_in_strings(payload), strict=False)


def _extract_json_array(text: str) -> list:
    """Parse a JSON array from a model response, tolerating prose/fences.

    Parse-first: the model is asked to return ONLY JSON, so try the raw text
    (tolerant of control chars) before touching fences. This matters because a
    file's content inside the JSON may itself contain a ``` code fence; stripping
    fences first would grab that inner fence and discard the real envelope. Only
    if the raw text is not itself JSON do we fall back to unwrapping an outer
    fence and slicing between the outer [ and ].
    """
    try:
        result = _loads_tolerant(text.strip())
        if isinstance(result, list):
            return result
    except (ValueError, json.JSONDecodeError):
        pass
    cleaned = _strip_fences(text)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in model response: {text[:300]}")
    return _loads_tolerant(cleaned[start : end + 1])


def _extract_json_object(text: str) -> dict:
    """Parse a JSON object from a model response, tolerating prose/fences.

    Parse-first (see ``_extract_json_array``): try the raw text before fence
    stripping so a ``` code fence *inside* a returned file's content cannot cause
    the whole JSON envelope to be thrown away. Fall back to outer-fence unwrap +
    slicing between the outer { and } only when the raw text is not itself JSON.
    """
    try:
        result = _loads_tolerant(text.strip())
        if isinstance(result, dict):
            return result
    except (ValueError, json.JSONDecodeError):
        pass
    cleaned = _strip_fences(text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model response: {text[:300]}")
    return _loads_tolerant(cleaned[start : end + 1])


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
- The response MUST be a single valid JSON object. Escape every newline inside a \
file's content as \\n, tabs as \\t, and quotes as \\" so the JSON parses.
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
