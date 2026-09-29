"""Full test campaign runner for the MITRA coherence check (issue #13, sub-task 8).

For every category in the saved conversation-starter selection (see
eval/starters.py; up to 3 random questions per category, 3 runs each) this
runner:

  1. Runs the question through the Reachy Mini agent (main.py, in the
     simulator). The first exchange is question -> agent reply. After that,
     GPT5.6 plays the human and writes a Sanskrit reply to the agent's last
     line, which is fed back to main.py, and so on for --max-turns agent turns.
  2. Runs the same seed question through GPT5.6 alone as a baseline, with the
     same GPT5.6 user simulator producing the follow-ups.
  3. Logs every spoken reply line with its depth through eval.harness
     (first line = depth 0), tagged with `system` = "reachy" or "gpt".
  4. Optionally runs a scorer callable over both record sets and writes a
     results JSON, and prints a per-category, per-depth summary.

This module does not define the grammar / Hindi verifiers or the score
formulas; plug them in with --grammar-fn, --hindi-fn and --scorer using
"package.module:function" specs (for example functions from eval.metrics,
eval.judged_metrics and eval.scoring).

Assumption about main.py: it reads one line of user text per turn from stdin
and logs each agent reply line with coherence.spoken_log.log_spoken_line
(see docs/coherence-check-setup.md). If your main.py takes input another way,
supply --agent-factory "package.module:function", a callable returning an
object with ask(text) -> list[str] and close().

Usage (from the mitra/ directory, with the simulator running and the judge
environment variables set):

    python -m eval.campaign --dry-run
    python -m eval.campaign --grammar-fn eval.metrics:grammar_score \\
        --hindi-fn eval.metrics:hindi_flag --scorer eval.scoring:score_campaign
    python -m eval.campaign --skip-baseline        # Reachy Mini only
    python -m eval.campaign --skip-reachy          # GPT5.6 baseline only

Completed (system, category, question, run) combinations are recorded in
logs/campaign_state.jsonl, so an interrupted campaign can be resumed by
re-running the same command. Delete the state file to start over.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from coherence.spoken_log import read_spoken_lines
from eval import starters
from eval.harness import Harness, load_records, summarize

MITRA_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = MITRA_DIR / "logs"
DEFAULT_REACHY_LOG = LOG_DIR / "campaign_reachy.jsonl"
DEFAULT_GPT_LOG = LOG_DIR / "campaign_gpt.jsonl"
DEFAULT_SPOKEN_LOG = LOG_DIR / "campaign_spoken.jsonl"
DEFAULT_STATE = LOG_DIR / "campaign_state.jsonl"
DEFAULT_RESULTS = LOG_DIR / "campaign_results.json"

GPT_PARTNER_PROMPT = (
    "You are a Sanskrit conversation partner. Reply only in Sanskrit "
    "(Devanagari script), in one or two short sentences. Do not use Hindi "
    "or English, and do not add translations."
)
GPT_USER_PROMPT = (
    "You are a learner having a friendly conversation in Sanskrit "
    "(Devanagari script). Reply to what your partner just said with one "
    "short Sanskrit sentence that keeps the conversation going. Output only "
    "the Sanskrit reply, with no Hindi, English, or translation."
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def resolve(spec):
    """Resolve 'package.module:function' to a callable (None if spec is falsy)."""
    if not spec:
        return None
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:function', got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def _question_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("question", "text", "q", "sanskrit"):
            if isinstance(item.get(key), str):
                return item[key]
    raise ValueError(f"cannot read a question from {item!r}")


def _questions_of(value):
    if isinstance(value, dict):
        for key in ("questions", "selected", "items"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError(f"cannot read question list from {value!r}")
    return [_question_text(v) for v in value]


def load_selection(path):
    """Return [(category, [questions])] from the saved starters JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("categories"), (dict, list)):
        data = data["categories"]
    out = []
    if isinstance(data, dict):
        for name, value in data.items():
            out.append((name, _questions_of(value)))
    elif isinstance(data, list):
        for entry in data:
            name = entry.get("name") or entry.get("category")
            out.append((name, _questions_of(entry)))
    else:
        raise ValueError("unrecognised starters file layout")
    cap = starters.MAX_QUESTIONS_PER_CATEGORY
    return [(name, qs[:cap]) for name, qs in out]


class State:
    """Append-only record of completed conversations, for resuming."""

    def __init__(self, path):
        self.path = Path(path)
        self.done = set()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                self.done = {line.strip() for line in f if line.strip()}

    @staticmethod
    def key(system, category, question, run):
        return json.dumps([system, category, question, run], ensure_ascii=False)

    def is_done(self, *args):
        return self.key(*args) in self.done

    def mark(self, *args):
        k = self.key(*args)
        self.done.add(k)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(k + "\n")


# --------------------------------------------------------------------------
# GPT5.6
# --------------------------------------------------------------------------

class GPT:
    """Thin chat wrapper; model and credentials come from the environment."""

    def __init__(self):
        from openai import OpenAI  # imported lazily so --dry-run needs no SDK

        model = os.environ.get("MITRA_JUDGE_MODEL")
        if not model or not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY and MITRA_JUDGE_MODEL must be set "
                "(run python -m coherence.check_env)."
            )
        self.model = model
        self.client = OpenAI()

    def chat(self, system, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return (resp.choices[0].message.content or "").strip()

    def user_followup(self, transcript):
        """Sanskrit reply from the simulated human, given [(speaker, text)]."""
        msgs = []
        for speaker, text in transcript:
            # From the simulated human's view, the agent is the "user".
            msgs.append({"role": "user" if speaker == "agent" else "assistant", "content": text})
        return self.chat(GPT_USER_PROMPT, msgs)


# --------------------------------------------------------------------------
# Reachy Mini agent (main.py in the simulator)
# --------------------------------------------------------------------------

class SubprocessAgent:
    """Runs main.py, feeds it lines on stdin and reads replies from the spoken log."""

    def __init__(self, spoken_log=DEFAULT_SPOKEN_LOG, timeout=90.0, settle=4.0, startup=10.0):
        self.spoken_log = Path(spoken_log)
        self.timeout = timeout
        self.settle = settle
        env = dict(os.environ, MITRA_SPOKEN_LOG=str(self.spoken_log))
        self.proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(MITRA_DIR),
            stdin=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
        time.sleep(startup)

    def _agent_lines(self):
        return [r["text"] for r in read_spoken_lines(self.spoken_log) if r.get("role") == "agent"]

    def ask(self, text):
        if self.proc.poll() is not None:
            raise RuntimeError("main.py exited unexpectedly")
        before = len(self._agent_lines())
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        start = time.time()
        last_change = None
        seen = before
        while time.time() - start < self.timeout:
            time.sleep(0.5)
            n = len(self._agent_lines())
            if n != seen:
                seen = n
                last_change = time.time()
            if last_change is not None and time.time() - last_change >= self.settle:
                break
        return self._agent_lines()[before:]

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


# --------------------------------------------------------------------------
# dialogues
# --------------------------------------------------------------------------

def run_reachy_dialogue(agent, gpt, conv, question, max_turns):
    """Question -> agent; then GPT5.6 replies drive further agent turns."""
    transcript = []
    user_text = question
    for turn in range(max_turns):
        lines = agent.ask(user_text)
        if not lines:
            print(f"    no agent reply at turn {turn}; ending dialogue")
            break
        for line in lines:
            conv.log_line(line, system="reachy", prompt=user_text, turn=turn)
        transcript.append(("user", user_text))
        transcript.append(("agent", " ".join(lines)))
        if turn == max_turns - 1:
            break
        user_text = gpt.user_followup(transcript)


def run_gpt_dialogue(gpt, conv, question, max_turns):
    """Same seed question through GPT5.6 alone; GPT5.6 also plays the human."""
    history = []
    transcript = []
    user_text = question
    for turn in range(max_turns):
        history.append({"role": "user", "content": user_text})
        reply = gpt.chat(GPT_PARTNER_PROMPT, history)
        history.append({"role": "assistant", "content": reply})
        conv.log_line(reply, system="gpt", prompt=user_text, turn=turn)
        transcript.append(("user", user_text))
        transcript.append(("agent", reply))
        if turn == max_turns - 1:
            break
        user_text = gpt.user_followup(transcript)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _print_summary(title, records):
    print(f"\n== {title} ==")
    if not records:
        print("(no records)")
        return
    print(f"{'category':<24} {'depth':>5} {'n':>4} {'grammar':>8} {'hindi':>6}")
    for row in summarize(records):
        g = "-" if row["mean_grammar_score"] is None else f"{row['mean_grammar_score']:.3f}"
        h = "-" if row["hindi_rate"] is None else f"{row['hindi_rate']:.3f}"
        print(f"{str(row['category']):<24} {row['depth']:>5} {row['n']:>4} {g:>8} {h:>6}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MITRA full coherence test campaign")
    p.add_argument("--starters", help="saved starters JSON (default: v1 selection)")
    p.add_argument("--runs", type=int, default=starters.RUNS_PER_QUESTION)
    p.add_argument("--max-turns", type=int, default=4, help="agent turns per dialogue")
    p.add_argument("--reachy-log", default=str(DEFAULT_REACHY_LOG))
    p.add_argument("--gpt-log", default=str(DEFAULT_GPT_LOG))
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.add_argument("--results", default=str(DEFAULT_RESULTS))
    p.add_argument("--grammar-fn", help="module:function(text) -> float")
    p.add_argument("--hindi-fn", help="module:function(text) -> bool")
    p.add_argument("--scorer", help="module:function(reachy_records, gpt_records) -> dict")
    p.add_argument("--agent-factory", help="module:function() -> object with ask(text)->list[str], close()")
    p.add_argument("--timeout", type=float, default=90.0)
    p.add_argument("--settle", type=float, default=4.0)
    p.add_argument("--skip-reachy", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args(argv)

    sel_path = args.starters or starters.output_path()
    selection = load_selection(sel_path)
    n_q = sum(len(qs) for _, qs in selection)
    n_conv = n_q * args.runs
    print(f"Selection: {sel_path}")
    print(f"{len(selection)} categories (expected {starters.EXPECTED_CATEGORIES}), "
          f"{n_q} questions, {args.runs} runs each = {n_conv} dialogues per system")
    if len(selection) != starters.EXPECTED_CATEGORIES:
        print("WARNING: category count differs from the expected 27.")
    if args.dry_run:
        for name, qs in selection:
            print(f"  {name}: {len(qs)} question(s)")
        return 0

    grammar_fn = resolve(args.grammar_fn)
    hindi_fn = resolve(args.hindi_fn)
    state = State(args.state)
    gpt = GPT()
    reachy_h = Harness(grammar_fn=grammar_fn, hindi_fn=hindi_fn, path=args.reachy_log)
    gpt_h = Harness(grammar_fn=grammar_fn, hindi_fn=hindi_fn, path=args.gpt_log)
    factory = resolve(args.agent_factory) or (
        lambda: SubprocessAgent(timeout=args.timeout, settle=args.settle)
    )

    for category, questions in selection:
        for question in questions:
            for run in range(1, args.runs + 1):
                if not args.skip_reachy and not state.is_done("reachy", category, question, run):
                    print(f"[reachy] {category} run {run}: {question}")
                    agent = factory()
                    try:
                        conv = reachy_h.conversation(category, question, run)
                        run_reachy_dialogue(agent, gpt, conv, question, args.max_turns)
                    finally:
                        agent.close()
                    state.mark("reachy", category, question, run)
                if not args.skip_baseline and not state.is_done("gpt", category, question, run):
                    print(f"[gpt]    {category} run {run}: {question}")
                    conv = gpt_h.conversation(category, question, run)
                    run_gpt_dialogue(gpt, conv, question, args.max_turns)
                    state.mark("gpt", category, question, run)

    reachy_records = load_records(args.reachy_log)
    gpt_records = load_records(args.gpt_log)
    _print_summary("Reachy Mini", reachy_records)
    _print_summary("GPT5.6 baseline", gpt_records)

    scorer = resolve(args.scorer)
    if scorer is not None:
        results = scorer(reachy_records, gpt_records)
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nWrote scores to {out}")
    else:
        print("\nNo --scorer given; component, coherence and purity scores not computed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())