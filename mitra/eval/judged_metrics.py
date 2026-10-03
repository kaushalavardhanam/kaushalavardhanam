"""GPT5.6-assisted judging for coherence metrics that cannot be computed by code.

Covers:
  * cross-turn repeats / parroting  (needs the previous turns of the dialogue)
  * Hindi-influenced forms in Sanskrit replies
  * any other metric with no programmatic calculation (custom rubric)

Every judgment gets the earlier turns of the same dialogue as context and must
answer with STRUCTURED JSON: integer counts plus concrete examples. Responses
that are not valid JSON or that miss required keys are retried once, then
recorded as an error (never silently dropped or guessed).

The judge is a plain callable ``judge_fn(prompt: str) -> str`` so it can be
swapped (tests use a fake). The default, ``default_judge``, calls an
OpenAI-compatible chat-completions endpoint using the credentials described in
mitra/docs/coherence-check-setup.md (OPENAI_API_KEY, MITRA_JUDGE_MODEL,
optional OPENAI_BASE_URL). It does not depend on the internals of
coherence/judge_client.py; pass ``judge_fn`` to use that client instead.

Usage (from the mitra/ directory):

    python -m eval.judged_metrics run                     # repeats + hindi over harness records
    python -m eval.judged_metrics run --metrics repeats
    python -m eval.judged_metrics run --custom-metric register \\
        --rubric "Count sentences that switch between formal and colloquial register."
    python -m eval.judged_metrics summary                 # per category/depth means
    python -m eval.judged_metrics sample --n 10           # draw spot-check sample
    python -m eval.judged_metrics report                  # agreement after manual review

Inputs : mitra/logs/harness_records.jsonl (MITRA_HARNESS_LOG)
Outputs: mitra/logs/judged_metrics.jsonl  (MITRA_JUDGED_LOG)
         mitra/logs/judged_spotcheck.jsonl (MITRA_SPOTCHECK_LOG)
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_JUDGED_PATH = LOG_DIR / "judged_metrics.jsonl"
DEFAULT_SPOTCHECK_PATH = LOG_DIR / "judged_spotcheck.jsonl"
SPOTCHECK_SEED = 13
MAX_ATTEMPTS = 2  # first try + one retry on malformed output

KIND_REPEATS = "repeats"
KIND_HINDI = "hindi_forms"
KIND_CUSTOM_PREFIX = "custom:"


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

_COMMON_RULES = """\
You are a careful evaluator of Sanskrit conversation quality for a robot
called MITRA. Judge ONLY the CURRENT REPLY, using the PREVIOUS TURNS purely as
context. Do not invent examples: every example must quote text that really
appears in the given turns. If there is nothing to report, return count 0 and
an empty examples list.

Respond with a single JSON object and NOTHING else (no prose, no code fences).
"""


def format_history(history):
    """Render previous turns as numbered lines (turn 0 is the first spoken line)."""
    if not history:
        return "(none - this is the first spoken line)"
    return "\n".join(f"[turn {i}] {t}" for i, t in enumerate(history))


def build_repeats_prompt(history, reply):
    return f"""{_COMMON_RULES}
TASK: cross-turn repetition and parroting.
Definitions:
- "repeat": the CURRENT REPLY reuses a sentence, clause or distinctive phrase
  that already appeared in an earlier turn of the agent, without adding
  anything new.
- "parroting": the CURRENT REPLY echoes the user's words back (a copied
  question or statement) instead of answering or advancing the dialogue.
Generic greetings/particles alone do not count.

PREVIOUS TURNS (oldest first):
{format_history(history)}

CURRENT REPLY:
{reply}

JSON schema:
{{
  "repeat_count": <integer >= 0>,
  "parroting_count": <integer >= 0>,
  "examples": [
    {{"type": "repeat" | "parroting",
      "span": "<exact text from the CURRENT REPLY>",
      "earlier_turn": <integer turn index it repeats/echoes>,
      "note": "<one short sentence>"}}
  ]
}}
repeat_count and parroting_count must equal the number of examples of that type.
"""


def build_hindi_prompt(history, reply):
    return f"""{_COMMON_RULES}
TASK: Hindi-influenced forms in a reply that is meant to be Sanskrit.
Count distinct Hindi-influenced items in the CURRENT REPLY, such as Hindi
vocabulary or particles (e.g. है, का, की, को, में, से, नहीं, और), Hindi
postpositions or auxiliary verbs, Hindi verb inflection, or Hindi word order
and calques where correct Sanskrit uses case endings or a different
construction. Correct Sanskrit words that merely look similar to Hindi do NOT
count. Count each distinct item once.

PREVIOUS TURNS (context only, do not score them):
{format_history(history)}

CURRENT REPLY:
{reply}

JSON schema:
{{
  "hindi_form_count": <integer >= 0>,
  "examples": [
    {{"span": "<exact text from the CURRENT REPLY>",
      "hindi_form": "<what Hindi feature it is>",
      "sanskrit_alternative": "<suggested Sanskrit form>",
      "note": "<one short sentence>"}}
  ]
}}
hindi_form_count must equal the number of examples.
"""


def build_custom_prompt(name, rubric, history, reply):
    return f"""{_COMMON_RULES}
TASK: custom metric "{name}".
Rubric:
{rubric}

PREVIOUS TURNS (oldest first):
{format_history(history)}

CURRENT REPLY:
{reply}

JSON schema:
{{
  "count": <integer >= 0, number of occurrences the rubric asks for>,
  "score": <number or null, only if the rubric defines a score>,
  "examples": [
    {{"span": "<exact text from the turns>", "note": "<one short sentence>"}}
  ]
}}
"""


# --------------------------------------------------------------------------
# Default judge (OpenAI-compatible chat completions)
# --------------------------------------------------------------------------

def default_judge(prompt):
    """Send the prompt to the configured judge model and return its text."""
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("MITRA_JUDGE_MODEL")
    if not api_key or not model:
        raise RuntimeError(
            "OPENAI_API_KEY and MITRA_JUDGE_MODEL must be set "
            "(see python -m coherence.check_env)"
        )
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# Parsing and validation
# --------------------------------------------------------------------------

class JudgeFormatError(ValueError):
    """The judge reply was not the required structured JSON."""


def extract_json(raw):
    """Parse the outermost JSON object in a model reply (tolerates code fences)."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        raise JudgeFormatError("no JSON object found in judge reply")
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeFormatError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise JudgeFormatError("judge reply JSON is not an object")
    return obj


def _require_count(obj, key):
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JudgeFormatError(f"'{key}' must be a non-negative integer, got {value!r}")
    return value


def _require_examples(obj):
    examples = obj.get("examples")
    if not isinstance(examples, list) or not all(isinstance(e, dict) for e in examples):
        raise JudgeFormatError("'examples' must be a list of objects")
    return examples


def validate_repeats(obj):
    obj["repeat_count"] = _require_count(obj, "repeat_count")
    obj["parroting_count"] = _require_count(obj, "parroting_count")
    examples = _require_examples(obj)
    warnings = []
    for kind, key in (("repeat", "repeat_count"), ("parroting", "parroting_count")):
        n = sum(1 for e in examples if e.get("type") == kind)
        if n != obj[key]:
            warnings.append(f"{key}={obj[key]} but {n} '{kind}' examples")
    if warnings:
        obj["warnings"] = warnings
    return obj


def validate_hindi(obj):
    obj["hindi_form_count"] = _require_count(obj, "hindi_form_count")
    examples = _require_examples(obj)
    if len(examples) != obj["hindi_form_count"]:
        obj["warnings"] = [
            f"hindi_form_count={obj['hindi_form_count']} but {len(examples)} examples"
        ]
    return obj


def validate_custom(obj):
    obj["count"] = _require_count(obj, "count")
    _require_examples(obj)
    score = obj.get("score")
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
        raise JudgeFormatError(f"'score' must be a number or null, got {score!r}")
    obj.setdefault("score", None)
    return obj


def _judge_structured(prompt, validator, judge_fn):
    """Call the judge, parse and validate; retry once on malformed output."""
    last_error = None
    for _ in range(MAX_ATTEMPTS):
        raw = judge_fn(prompt)
        try:
            return validator(extract_json(raw))
        except JudgeFormatError as exc:
            last_error = exc
            prompt = (
                prompt
                + f"\n\nYour previous reply was rejected ({exc}). "
                "Reply with ONLY the JSON object in the required schema."
            )
    raise last_error


def judge_repeats(history, reply, judge_fn=None):
    """Structured cross-turn repeat / parroting counts with examples."""
    return _judge_structured(
        build_repeats_prompt(history, reply), validate_repeats, judge_fn or default_judge
    )


def judge_hindi_forms(history, reply, judge_fn=None):
    """Structured count of Hindi-influenced forms with examples."""
    return _judge_structured(
        build_hindi_prompt(history, reply), validate_hindi, judge_fn or default_judge
    )


def judge_custom(name, rubric, history, reply, judge_fn=None):
    """Structured judgment for a metric that has no programmatic calculation."""
    return _judge_structured(
        build_custom_prompt(name, rubric, history, reply),
        validate_custom,
        judge_fn or default_judge,
    )


# --------------------------------------------------------------------------
# Running over harness records
# --------------------------------------------------------------------------

def _judged_path(path=None):
    if path:
        return Path(path)
    override = os.environ.get("MITRA_JUDGED_LOG")
    return Path(override) if override else DEFAULT_JUDGED_PATH


def _spotcheck_path(path=None):
    if path:
        return Path(path)
    override = os.environ.get("MITRA_SPOTCHECK_LOG")
    return Path(override) if override else DEFAULT_SPOTCHECK_PATH


def _read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_judged(path=None):
    return _read_jsonl(_judged_path(path))


def _key(rec, kind):
    return (kind, rec.get("category"), rec.get("seed_question"), rec.get("run"), rec.get("depth"))


def group_dialogues(records):
    """Group harness records into dialogues, each sorted by depth."""
    groups = defaultdict(list)
    for r in records:
        groups[(r.get("category"), r.get("seed_question"), r.get("run"))].append(r)
    return {k: sorted(v, key=lambda r: r.get("depth", 0)) for k, v in groups.items()}


def previous_turns(dialogue, index):
    return [r["text"] for r in dialogue[:index]]


def run_judging(records, kinds, judge_fn=None, out_path=None, customs=None, force=False):
    """Judge every record for each requested kind and append to the judged log.

    kinds:   subset of {"repeats", "hindi"}
    customs: optional dict {metric_name: rubric}
    Already judged (kind, dialogue, depth) entries are skipped unless force.
    Returns the list of newly written result records.
    """
    judge_fn = judge_fn or default_judge
    out = _judged_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if not force:
        for r in _read_jsonl(out):
            if not r.get("error"):
                done.add(_key(r, r.get("kind")))

    jobs = []
    if "repeats" in kinds:
        jobs.append((KIND_REPEATS, lambda h, t: judge_repeats(h, t, judge_fn)))
    if "hindi" in kinds:
        jobs.append((KIND_HINDI, lambda h, t: judge_hindi_forms(h, t, judge_fn)))
    for name, rubric in (customs or {}).items():
        jobs.append(
            (KIND_CUSTOM_PREFIX + name, lambda h, t, n=name, rb=rubric: judge_custom(n, rb, h, t, judge_fn))
        )

    written = []
    with out.open("a", encoding="utf-8") as f:
        for dialogue in group_dialogues(records).values():
            for i, rec in enumerate(dialogue):
                history = previous_turns(dialogue, i)
                for kind, fn in jobs:
                    if _key(rec, kind) in done:
                        continue
                    entry = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "kind": kind,
                        "category": rec.get("category"),
                        "seed_question": rec.get("seed_question"),
                        "run": rec.get("run"),
                        "depth": rec.get("depth"),
                        "text": rec.get("text"),
                        "history": history,
                        "model": os.environ.get("MITRA_JUDGE_MODEL"),
                    }
                    try:
                        entry["result"] = fn(history, rec["text"])
                    except Exception as exc:  # keep going; record the failure
                        entry["error"] = f"{type(exc).__name__}: {exc}"
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    written.append(entry)
    return written


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def _primary_count(entry):
    result = entry.get("result") or {}
    kind = entry.get("kind")
    if kind == KIND_REPEATS:
        return result.get("repeat_count", 0) + result.get("parroting_count", 0)
    if kind == KIND_HINDI:
        return result.get("hindi_form_count")
    return result.get("count")


def summarize_judged(entries):
    """Per (kind, category, depth): n judged, n errors, mean count."""
    groups = defaultdict(list)
    for e in entries:
        groups[(e.get("kind"), e.get("category"), e.get("depth"))].append(e)
    rows = []
    for (kind, category, depth), es in sorted(
        groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), kv[0][2] if kv[0][2] is not None else -1)
    ):
        counts = [_primary_count(e) for e in es if not e.get("error")]
        counts = [c for c in counts if c is not None]
        rows.append(
            {
                "kind": kind,
                "category": category,
                "depth": depth,
                "n": len(es),
                "errors": sum(1 for e in es if e.get("error")),
                "mean_count": sum(counts) / len(counts) if counts else None,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Manual spot check
# --------------------------------------------------------------------------

def draw_spotcheck(entries, n=10, seed=SPOTCHECK_SEED, out_path=None):
    """Randomly sample judged entries (fixed seed) into a file for manual review.

    Each sampled row gets blank ``manual_agree`` (true/false) and ``manual_note``
    fields to be filled in by hand after reading the turn and the judge's
    examples. Returns the sampled rows.
    """
    ok = [e for e in entries if not e.get("error")]
    rng = random.Random(seed)
    sample = rng.sample(ok, min(n, len(ok)))
    rows = []
    for e in sample:
        row = dict(e)
        row["manual_agree"] = None
        row["manual_note"] = ""
        rows.append(row)
    path = _spotcheck_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def spotcheck_report(path=None):
    """Agreement between the judge and the manual reviewer."""
    rows = _read_jsonl(_spotcheck_path(path))
    reviewed = [r for r in rows if r.get("manual_agree") is not None]
    agree = sum(1 for r in reviewed if r["manual_agree"])
    return {
        "sampled": len(rows),
        "reviewed": len(reviewed),
        "agree": agree,
        "agreement_rate": agree / len(reviewed) if reviewed else None,
    }


def _print_sample(rows):
    for i, r in enumerate(rows, 1):
        print(f"--- sample {i}: {r['kind']} | {r['category']} | run {r['run']} | depth {r['depth']}")
        for j, t in enumerate(r.get("history") or []):
            print(f"  [turn {j}] {t}")
        print(f"  CURRENT: {r['text']}")
        print("  JUDGE:", json.dumps(r["result"], ensure_ascii=False))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    from eval.harness import load_records

    parser = argparse.ArgumentParser(description="GPT5.6-assisted coherence metrics")
    parser.add_argument("--harness-path", help="harness JSONL (default: env or logs/)")
    parser.add_argument("--judged-path", help="judged JSONL (default: env or logs/)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="judge harness records")
    run_p.add_argument("--metrics", nargs="+", choices=["repeats", "hindi"], default=["repeats", "hindi"])
    run_p.add_argument("--custom-metric", help="name of an extra hand-judged metric")
    run_p.add_argument("--rubric", help="rubric text for --custom-metric")
    run_p.add_argument("--force", action="store_true", help="re-judge already judged records")

    sub.add_parser("summary", help="print per kind/category/depth means")

    sample_p = sub.add_parser("sample", help="draw a manual spot-check sample")
    sample_p.add_argument("--n", type=int, default=10)
    sample_p.add_argument("--seed", type=int, default=SPOTCHECK_SEED)

    sub.add_parser("report", help="judge/manual agreement after review")
    args = parser.parse_args(argv)

    if args.cmd == "run":
        customs = None
        if args.custom_metric:
            if not args.rubric:
                parser.error("--custom-metric requires --rubric")
            customs = {args.custom_metric: args.rubric}
        records = load_records(args.harness_path)
        if not records:
            print("No harness records found.")
            return 1
        written = run_judging(
            records, set(args.metrics), out_path=args.judged_path, customs=customs, force=args.force
        )
        errors = sum(1 for w in written if w.get("error"))
        print(f"Judged {len(written)} items ({errors} errors) -> {_judged_path(args.judged_path)}")
        return 1 if errors else 0

    if args.cmd == "summary":
        entries = load_judged(args.judged_path)
        if not entries:
            print("No judged records at", _judged_path(args.judged_path))
            return 0
        print(f"{'kind':<20} {'category':<20} {'depth':>5} {'n':>4} {'err':>4} {'mean':>7}")
        for r in summarize_judged(entries):
            mean = "-" if r["mean_count"] is None else f"{r['mean_count']:.3f}"
            print(
                f"{str(r['kind']):<20} {str(r['category']):<20} {r['depth']:>5} "
                f"{r['n']:>4} {r['errors']:>4} {mean:>7}"
            )
        return 0

    if args.cmd == "sample":
        entries = load_judged(args.judged_path)
        if not entries:
            print("No judged records to sample.")
            return 1
        rows = draw_spotcheck(entries, n=args.n, seed=args.seed)
        _print_sample(rows)
        print(f"\nWrote {len(rows)} rows to {_spotcheck_path()}.")
        print("Set manual_agree (true/false) and manual_note in each row, then run: report")
        return 0

    if args.cmd == "report":
        rep = spotcheck_report()
        rate = "-" if rep["agreement_rate"] is None else f"{rep['agreement_rate']:.2f}"
        print(
            f"sampled={rep['sampled']} reviewed={rep['reviewed']} "
            f"agree={rep['agree']} agreement_rate={rate}"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())