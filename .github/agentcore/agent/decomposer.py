"""LangGraph decompose-then-implement workflow for the AgentCore agent.

This builds on the multi-agent LangGraph pattern in the repo's
``coding-agent/graph_workflow.py`` (a supervisor node that routes between
worker nodes via a shared ``WorkflowState`` TypedDict and a conditional-edge
router). Here the workflow is specialised for the issue-to-PR pipeline:

    decompose  ->  implement (loops over sub-tasks)  ->  END

* **decompose**: one Bedrock call turns the GitHub issue into an *ordered* list
  of sub-tasks. This stays a strict JSON array because it carries only
  lightweight metadata (id / title / description) — never file bodies.
* **implement**: for each sub-task in order, one Bedrock call produces the set
  of file writes for that step. The file CONTENTS are returned as RAW text
  between unique sentinel markers, NOT as JSON string values:

      <<<FILE path="relative/path.py">>>
      ...verbatim file body, no escaping...
      <<<END FILE>>>

  This is the durable fix for issue #13. Embedding whole file contents as JSON
  string values made the implement response fragile in two structural ways:
    1. stochastic mis-escaping — on a large/complex file the model would
       occasionally emit an unescaped ``"`` / backslash / control char inside a
       string value, making the whole envelope unparseable (a content quote is
       indistinguishable from a string terminator, so no post-hoc parser can
       recover it);
    2. truncation blast radius — one monolithic JSON blob carrying every file
       could blow past the output-token cap mid-string, failing the entire
       sub-task.
  With the sentinel contract, file bodies are never JSON-escaped, so quotes /
  newlines / control chars in content can no longer break parsing, and a file
  truncated at the token cap is detectable (its ``<<<END FILE>>>`` sentinel is
  missing) and can be re-asked per-file and stitched, instead of failing the
  whole step.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph

from bedrock_client import BedrockClient, TruncatedResponseError

logger = logging.getLogger(__name__)

MAX_SUBTASKS = 12

# How many times a single truncated file body may be re-asked (continuation)
# before we give up on that file. Bounded so a pathologically un-terminating
# generation cannot loop forever.
MAX_FILE_CONTINUATIONS = 4


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
# JSON extraction helpers (decompose metadata only — never file bodies)
# --------------------------------------------------------------------------- #

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Return the content of the first ```...``` fence, or the text unchanged."""
    match = _JSON_FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def _extract_json_array(text: str) -> list:
    """Parse a JSON array from a model response, tolerating prose/fences.

    Used ONLY for the decompose step, whose payload is small structured
    metadata (no file contents), so strict JSON is appropriate here. Parse-first
    (try the raw text before unwrapping a fence); fall back to slicing between
    the outer ``[`` and ``]``.
    """
    stripped = text.strip()
    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    cleaned = _strip_fences(text)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in model response: {text[:300]}")
    return json.loads(cleaned[start : end + 1])


# --------------------------------------------------------------------------- #
# Sentinel-delimited file parsing (implement step — verbatim file bodies)
# --------------------------------------------------------------------------- #

# A file block opens with a header line naming its repo-relative path and closes
# with a standalone end sentinel. Everything between (exclusive of the header's
# trailing newline and the end sentinel) is the file body, taken VERBATIM.
#
#   <<<FILE path="relative/path.py">>>
#   ...body...
#   <<<END FILE>>>
#
# The header is matched with the path in double quotes; a bare form
# ``<<<FILE relative/path.py>>>`` is also accepted for robustness. The end
# sentinel must sit on its own line. Regexes are deliberately anchored to line
# starts (MULTILINE) so a sentinel-looking string INSIDE a file body (unlikely,
# but possible) is far less likely to false-match — and even if content
# contained the literal marker, that is content the model itself chose to emit
# and no escaping scheme is involved.
_FILE_OPEN = re.compile(
    r'^<<<FILE\s+(?:path\s*=\s*"(?P<q>[^"\n]+)"|(?P<b>[^\n>]+?))\s*>>>[ \t]*\r?\n',
    re.MULTILINE,
)
_FILE_END = re.compile(r'^<<<END FILE>>>[ \t]*\r?$', re.MULTILINE)
# An optional one-line note the model may emit OUTSIDE any file block.
_NOTE_LINE = re.compile(r'^<<<NOTE>>>[ \t]*(?P<note>.*?)[ \t]*<<<END NOTE>>>',
                        re.MULTILINE | re.DOTALL)


class _ParsedFile:
    """One file parsed from a sentinel-delimited response."""

    __slots__ = ("path", "body", "truncated")

    def __init__(self, path: str, body: str, truncated: bool) -> None:
        self.path = path
        self.body = body
        self.truncated = truncated


def _parse_sentinel_files(text: str) -> Tuple[List[_ParsedFile], Optional[str]]:
    """Split a sentinel-delimited implement response into files + an optional note.

    Returns ``(files, note)``. Each file's body is taken VERBATIM between its
    ``<<<FILE ...>>>`` header line and the next ``<<<END FILE>>>`` sentinel — no
    JSON parsing, no unescaping, so quotes/newlines/control chars in content are
    harmless. A file whose closing sentinel is MISSING before the next file
    header (or EOF) is marked ``truncated`` (its body was cut off at the token
    cap); the caller re-asks just that file rather than failing the whole step.
    """
    files: List[_ParsedFile] = []

    opens = list(_FILE_OPEN.finditer(text))
    for idx, m in enumerate(opens):
        path = (m.group("q") or m.group("b") or "").strip()
        if not path:
            continue
        body_start = m.end()
        # The body runs until the FIRST end sentinel that appears before the
        # NEXT file header (so a missing end sentinel is detected rather than
        # greedily swallowing the following file).
        next_open_start = opens[idx + 1].start() if idx + 1 < len(opens) else len(text)
        end_m = _FILE_END.search(text, body_start, next_open_start)
        if end_m is not None:
            body = text[body_start : end_m.start()]
            truncated = False
        else:
            # No closing sentinel before the next header / EOF => truncated.
            body = text[body_start:next_open_start]
            truncated = True
        # Strip exactly one trailing newline that precedes the end sentinel /
        # boundary (the header line and the end sentinel each own their own
        # newline); preserve all interior content verbatim.
        if body.endswith("\r\n"):
            body = body[:-2]
        elif body.endswith("\n"):
            body = body[:-1]
        files.append(_ParsedFile(path=path, body=body, truncated=truncated))

    note_m = _NOTE_LINE.search(text)
    note = note_m.group("note").strip() if note_m else None
    return files, note


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
    "conventions of the repository. You emit each file's raw content between the "
    "required sentinel markers exactly as instructed — never as JSON, never "
    "escaped."
)

# The output-contract block, shared by the initial implement prompt and the
# per-file continuation prompt so the model sees identical framing.
_SENTINEL_CONTRACT = """\
Output format — emit each file EXACTLY like this, with the file's content RAW \
and VERBATIM between the markers (do NOT wrap it in JSON, do NOT escape quotes, \
newlines, or any character):

<<<FILE path="repo-relative/posix/path.ext">>>
<the complete file content, exactly as it should appear on disk>
<<<END FILE>>>

Then, after all files, optionally emit a single note line:
<<<NOTE>>>one-line summary of what you did<<<END NOTE>>>

Rules:
- One <<<FILE ...>>> ... <<<END FILE>>> block per file you create or change.
- The path is a repo-relative POSIX path. Do NOT use absolute paths or '..'.
- Put the COMPLETE new content of each file between the markers, not a diff.
- Emit NOTHING else outside the file blocks and the optional note (no prose, no \
markdown fences).
- The <<<END FILE>>> marker MUST be on its own line. If you run out of room, \
still finish the file you are on before stopping — a file without its \
<<<END FILE>>> will be treated as truncated."""


def _build_implement_prompt(state: DecomposeState, task: SubTask,
                            existing: str, existing_context: str) -> str:
    return f"""You are implementing ONE sub-task for issue \
#{state.get('issue_number')} ("{state.get('title')}") in repo \
{state.get('repo')}.

Sub-task {task['id']}: {task['title']}
{task['description']}

Files already generated by earlier sub-tasks:
{existing}

Contents of those files (for consistency; you may return updated versions):
{existing_context or '(none)'}

{_SENTINEL_CONTRACT}"""


def _build_continuation_prompt(state: DecomposeState, task: SubTask,
                               path: str, partial_body: str) -> str:
    """Prompt to resume a single file whose body was truncated at the token cap.

    Sends back the tail of what we have so the model continues EXACTLY from the
    cut point, and asks for ONLY that one file's remaining content — never
    re-emitting the whole file, so the continuation itself fits in the window.
    """
    tail = partial_body[-2000:]
    return f"""While implementing sub-task {task['id']} for issue \
#{state.get('issue_number')} in repo {state.get('repo')}, the file below was cut \
off because the response hit the output-token limit. Continue it.

File path: {path}

Here is the content generated so far (the END of it — continue from exactly \
where it stops, do NOT repeat any of it and do NOT restart the file):
<<<PARTIAL SO FAR>>>
{tail}
<<<END PARTIAL>>>

Emit ONLY the REMAINING content of this one file, raw and verbatim, between the \
markers — no JSON, no escaping, no prose:

<<<FILE path="{path}">>>
<the remaining content that comes AFTER the partial above>
<<<END FILE>>>

If the file was already complete at the cut point, emit an empty body between \
the markers."""


def _implement_node(bedrock: BedrockClient):
    def implement(state: DecomposeState) -> Dict:
        subtasks = state.get("subtasks", [])
        files: Dict[str, str] = dict(state.get("files", {}))
        notes: List[str] = list(state.get("implementation_notes", []))

        for task in subtasks:
            logger.info("Implementing sub-task %s: %s", task["id"], task["title"])

            existing = "\n".join(sorted(files)) or "(none yet)"
            existing_context = ""
            if files:
                existing_context = "\n\n".join(
                    f"--- {path} ---\n{content}" for path, content in files.items()
                )[:20000]

            prompt = _build_implement_prompt(state, task, existing, existing_context)

            # Use the non-raising variant: a truncated response is NOT a failure
            # here — completed file blocks are salvaged and any truncated file is
            # continued per-file below.
            raw, _truncated = bedrock.invoke_partial(
                prompt, system=IMPLEMENT_SYSTEM, temperature=0.2
            )

            parsed_files, note = _parse_sentinel_files(raw)
            if not parsed_files:
                # No sentinel blocks at all — the model ignored the contract, or
                # returned something unusable. This is a genuine step failure.
                logger.error(
                    "Sub-task %s produced no file blocks (response head: %s)",
                    task["id"], raw[:300],
                )
                return {
                    "error": (
                        f"sub-task {task['id']} produced no <<<FILE>>> blocks; "
                        f"response head: {raw[:200]!r}"
                    ),
                    "files": files,
                    "implementation_notes": notes,
                }

            for pf in parsed_files:
                body = pf.body
                if pf.truncated:
                    body = _continue_truncated_file(
                        bedrock, state, task, pf.path, body
                    )
                safe = _sanitize_path(pf.path)
                if safe is None:
                    logger.warning("Skipping unsafe path from model: %s", pf.path)
                    continue
                files[safe] = body

            notes.append(
                f"{task['id']}. {task['title']} — {note or 'implemented'}"
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


def _continue_truncated_file(
    bedrock: BedrockClient,
    state: DecomposeState,
    task: SubTask,
    path: str,
    partial_body: str,
) -> str:
    """Re-ask a single truncated file's remaining content and stitch it on.

    Loops (bounded by ``MAX_FILE_CONTINUATIONS``) because a very large file may
    need more than one continuation. Each round asks ONLY for the remaining
    content of this one file, so no single continuation call needs the full
    output window.
    """
    body = partial_body
    for attempt in range(1, MAX_FILE_CONTINUATIONS + 1):
        logger.info(
            "Continuing truncated file %s (sub-task %s, attempt %d)",
            path, task["id"], attempt,
        )
        cont_prompt = _build_continuation_prompt(state, task, path, body)
        raw, truncated = bedrock.invoke_partial(
            cont_prompt, system=IMPLEMENT_SYSTEM, temperature=0.2
        )
        cont_files, _note = _parse_sentinel_files(raw)
        # Take the block matching this path if present, else the first block.
        chunk = None
        for cf in cont_files:
            if _sanitize_path(cf.path) == _sanitize_path(path):
                chunk = cf
                break
        if chunk is None and cont_files:
            chunk = cont_files[0]
        if chunk is None:
            logger.warning(
                "Continuation for %s produced no file block; stopping stitch.", path
            )
            break
        body += chunk.body
        if not chunk.truncated and not truncated:
            # This continuation completed the file.
            return body
        # Still truncated — loop again to fetch the next segment.
    logger.warning(
        "File %s still truncated after %d continuation attempt(s); "
        "using best-effort stitched content.",
        path, MAX_FILE_CONTINUATIONS,
    )
    return body


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
