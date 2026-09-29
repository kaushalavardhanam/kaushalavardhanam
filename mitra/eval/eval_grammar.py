#!/usr/bin/env python3
"""
Score a Mitra run against the 50-question evaluation set.

    # print the 50 prompts, in order, to drive a run
    python3 eval_grammar.py --list

    # score a finished run
    python3 eval_grammar.py --turns logs/turns.jsonl

    # score and save a baseline, then compare later runs against it
    python3 eval_grammar.py --turns logs/turns.jsonl --save-baseline eval/baseline.json
    python3 eval_grammar.py --turns logs/turns.jsonl --baseline eval/baseline.json

Every check is deterministic string and morphology matching, so the same
turns.jsonl always yields the same score and a fix can be verified in one run.
Nothing here calls a model.

Exit status is 0 unless --baseline is given and the run regressed.
"""

import argparse
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- word lists

# Hindi forms seen in runs A to D, plus the common function words that would
# signal the model dropped out of Sanskrit entirely.
HINDI_MARKERS = [
    "खेलानि", "खेलनी", "खेलनीय", "खेल",
    "घरे", "घरम्", "घर",
    "आज",
    "मक्खन", "मक्खनम्", "मक्खनं",
    "दूध", "दूधम्", "दूधं",
    "दाल", "दालः",
    "तुम्हा", "तुम्हारा",
    "नहीं", "क्या", "कैसे", "है", "में", "और",
]

# Forms produced by the model that are not Sanskrit words at all.
NONWORDS = [
    "करोष्यसि", "करोष्यथ", "कुरुमि", "पठमि", "खेलनी",
    "अभवति", "पसिला", "मेरुद्वारकं", "मेरुद्वारक",
]

FIRST_PERSON = re.compile(r"(ामि|ोमि|्मि|ावः|ामः|ाव|ाम)[ ।?]|अस्मि")
THIRD_PERSON = re.compile(r"\S*(ति|न्ति|ते)[ ।?]")
SECOND_PERSON = re.compile(r"\S*(सि|थ)[ ।?]")
FUTURE = re.compile(r"ष्य(ामि|ति|सि|न्ति|थ)|श्वः")
PAST_MARKERS = ["ह्यः", "तवान्", "तवती", "अकर", "अगच्छ", "अपठ", "अखाद", "आसम्", "आसीत्"]
GERUNDIVE = re.compile(r"(नीयम्|तव्यम्|णीयम्|नीयं|तव्यं)")
SANDHI = re.compile(r"ं\s+[अआइईउऊऋएऐओऔ]")
MASC_SELF = re.compile(r"(?<![भकयतअ])\S*(तः|वान्)[ ।?]")
MASC_SELF_SKIP = {"भवतः", "ततः", "अतः", "यतः", "कुतः", "प्रातः"}

POSSESSIVE_OK = re.compile(r"मम\s.*अस्ति|मह्यं\s.*रोचते|मह्यम्\s.*रोचते")
POSSESSIVE_BROKEN = re.compile(r"अहं\s+\S+\s*प्रिय(ं|म्)?\s*(अस्मि)?")

# Failure classes that make a reply unclean.
ERROR_CLASSES = [
    "hindi_marker", "nonword", "person_mismatch", "frame_broken",
    "no_content", "forbidden_word", "tense_mismatch", "gerundive",
]
# Recorded and reported, but do not by themselves fail a reply.
WARN_CLASSES = ["sandhi", "persona_gender"]


# ---------------------------------------------------------------- utilities

def norm(s):
    """Normalise an English prompt for fuzzy matching against ASR output."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def clauses(reply):
    """Split a reply into the answer clause and anything after it."""
    parts = [p.strip() for p in re.split(r"[।?]", reply or "") if p.strip()]
    return parts or [""]


def pad(reply):
    """Trailing space so the end-anchored patterns match the last token."""
    return (reply or "").replace("।", " । ").replace("?", " ? ") + " "


# ---------------------------------------------------------------- the checks

def check_reply(q, reply):
    """Return (errors, warnings) as lists of (class, detail)."""
    errors, warns = [], []
    text = pad(reply)
    answer = clauses(reply)[0]
    answer_p = pad(answer)

    for w in HINDI_MARKERS:
        if w in reply:
            errors.append(("hindi_marker", w))
            break

    for w in NONWORDS:
        if w in reply:
            errors.append(("nonword", w))
            break

    for w in q.get("forbid", []) or []:
        if w in reply:
            errors.append(("forbidden_word", w))
            break

    # content: did it actually answer?
    hits = [w for w in q.get("content_any", []) or [] if w in reply]
    if q.get("content_any") and not hits:
        errors.append(("no_content", "none of the expected answer words"))

    # possessive frame
    if q.get("frame") == "possessive":
        if POSSESSIVE_BROKEN.search(reply or "") and not POSSESSIVE_OK.search(reply or ""):
            errors.append(("frame_broken", "अहं X प्रियं अस्मि"))

    # person agreement, judged on the answer clause only
    expected = q.get("person", "first")
    if expected == "first" and "अहं" in answer:
        third = THIRD_PERSON.search(answer_p)
        if third and not POSSESSIVE_OK.search(answer or ""):
            tok = third.group(0).strip()
            if tok not in ("अस्ति", "रोचते"):
                errors.append(("person_mismatch", f"अहं with {tok}"))
    if "भवान्" in text or "भवती" in text:
        m = SECOND_PERSON.search(text) or FIRST_PERSON.search(text)
        # भवान् takes a third person verb; a second person ending is the classic miss
        if SECOND_PERSON.search(text):
            errors.append(("person_mismatch", f"भवान् with {m.group(0).strip()}"))
    if "त्वम्" in text and re.search(r"अस्मि|ामि[ ।?]", text):
        errors.append(("person_mismatch", "त्वम् with a first person verb"))

    # tense
    tense = q.get("tense", "present")
    if tense == "future" and not FUTURE.search(reply or ""):
        errors.append(("tense_mismatch", "future question answered without a future"))
    if tense == "present" and FUTURE.search(reply or ""):
        errors.append(("tense_mismatch", "present question answered in the future"))
    if tense == "past" and not any(m in (reply or "") for m in PAST_MARKERS):
        errors.append(("tense_mismatch", "past question answered without a past"))

    if GERUNDIVE.search(reply or ""):
        errors.append(("gerundive", GERUNDIVE.search(reply).group(0)))

    if SANDHI.search(reply or ""):
        warns.append(("sandhi", "anusvara before a vowel, expected म्"))
    if "अहं" in reply:
        for m in MASC_SELF.finditer(answer_p):
            if m.group(0).strip() not in MASC_SELF_SKIP:
                warns.append(("persona_gender",
                              f"masculine {m.group(0).strip()}, Mitra is मित्रम्"))
                break

    return errors, warns


# ---------------------------------------------------------------- run loading

def load_turns(path):
    turns = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # tolerate raw log lines that embed the JSON after a "turn: " prefix
        i = line.find("{")
        if i < 0:
            continue
        try:
            turns.append(json.loads(line[i:]))
        except json.JSONDecodeError:
            continue
    return turns


def match_turns(questions, turns):
    """Pair each question with a turn, by question_id when present else fuzzily.

    Fuzzy pairing is a global greedy assignment: every question/turn similarity
    is scored, the strongest pair is taken first, and both sides are then
    retired. Assigning in question order instead would let an early question
    claim a turn that belongs to a later one.
    """
    by_id = {t["question_id"]: t for t in turns if t.get("question_id")}
    pool = [t for t in turns if not t.get("question_id")]
    need = [q for q in questions if q["id"] not in by_id]

    scored = []
    for qi, q in enumerate(need):
        for ti, t in enumerate(pool):
            r = difflib.SequenceMatcher(
                None, norm(q["text"]), norm(t.get("transcript"))).ratio()
            if r >= 0.55:
                scored.append((r, qi, ti))
    scored.sort(reverse=True)

    taken_q, taken_t, assign = set(), set(), {}
    for r, qi, ti in scored:
        if qi in taken_q or ti in taken_t:
            continue
        taken_q.add(qi)
        taken_t.add(ti)
        assign[need[qi]["id"]] = pool[ti]

    return [(q, by_id.get(q["id"]) or assign.get(q["id"])) for q in questions]


def stage_stats(turns):
    out = {}
    for stage in ("asr", "llm", "tts"):
        vals = sorted(t.get("stages", {}).get(stage, 0.0) for t in turns
                      if t.get("stages", {}).get(stage))
        if vals:
            out[stage] = {
                "p50": round(vals[len(vals) // 2], 2),
                "min": round(vals[0], 2),
                "max": round(vals[-1], 2),
            }
    return out


# ---------------------------------------------------------------- scoring

def score(questions, turns):
    pairs = match_turns(questions, turns)
    rows, classes = [], Counter()
    by_cat = defaultdict(lambda: [0, 0])
    replies_by_pair = defaultdict(list)
    asked = 0

    for q, t in pairs:
        if t is None:
            rows.append({"id": q["id"], "category": q["category"], "status": "missing",
                         "reply": "", "errors": [], "warnings": []})
            continue
        asked += 1
        reply = t.get("reply", "")
        errors, warns = check_reply(q, reply)
        for c, _ in errors:
            classes[c] += 1
        for c, _ in warns:
            classes[c] += 1
        clean = not errors
        by_cat[q["category"]][0] += int(clean)
        by_cat[q["category"]][1] += 1
        if q.get("pair"):
            replies_by_pair[q["pair"]].append((q["id"], reply))
        rows.append({
            "id": q["id"], "category": q["category"],
            "status": "clean" if clean else "flagged",
            "reply": reply,
            "errors": [f"{c}: {d}" for c, d in errors],
            "warnings": [f"{c}: {d}" for c, d in warns],
            "template_question_back": "?" in reply,
        })

    collapses = []
    for pair, items in replies_by_pair.items():
        if len(items) > 1 and len({r for _, r in items}) == 1:
            collapses.append({"pair": pair, "ids": [i for i, _ in items],
                              "reply": items[0][1]})
    classes["collapse"] = len(collapses)

    clean = sum(1 for r in rows if r["status"] == "clean")
    template = sum(1 for r in rows if r.get("template_question_back"))

    return {
        "asked": asked,
        "total": len(questions),
        "clean": clean,
        "clean_rate": round(clean / asked, 3) if asked else 0.0,
        "template_question_back": template,
        "classes": dict(classes),
        "by_category": {k: {"clean": v[0], "asked": v[1]} for k, v in sorted(by_cat.items())},
        "collapses": collapses,
        "latency": stage_stats(turns),
        "rows": rows,
    }


# ---------------------------------------------------------------- reporting

def report(res, verbose=False):
    print(f"\nMitra grammar eval  {res['clean']}/{res['asked']} clean "
          f"({res['clean_rate']:.0%})   [{res['asked']}/{res['total']} questions answered]")

    print("\nby category")
    for cat, v in res["by_category"].items():
        print(f"  {cat:<12} {v['clean']}/{v['asked']}")

    if res["classes"]:
        print("\nfailures by class")
        for c, n in sorted(res["classes"].items(), key=lambda kv: -kv[1]):
            if n:
                tag = "warn" if c in WARN_CLASSES else "    "
                print(f"  {tag} {c:<18} {n}")

    print(f"\ntemplate: {res['template_question_back']}/{res['asked']} replies ask a question back")

    for c in res["collapses"]:
        print(f"collapse: {' and '.join(c['ids'])} returned the identical reply")

    if res["latency"]:
        lat = res["latency"]
        line = "  ".join(f"{k} p50 {v['p50']}s ({v['min']}-{v['max']})"
                         for k, v in lat.items())
        print(f"\nlatency  {line}")

    flagged = [r for r in res["rows"] if r["status"] == "flagged"]
    if flagged:
        print("\nflagged replies")
        for r in flagged:
            print(f"  {r['id']}  {r['reply']}")
            for e in r["errors"]:
                print(f"        {e}")
            if verbose:
                for w in r["warnings"]:
                    print(f"        warn {w}")

    missing = [r["id"] for r in res["rows"] if r["status"] == "missing"]
    if missing:
        print(f"\nno turn matched: {', '.join(missing)}")


def aggregate(results):
    """Pool several runs of the same code into one result.

    A single run is not a measurement. Across six runs of near-identical code
    the clean rate spanned 58 to 68 percent, and one run showed nine
    person_mismatch where every other showed none or one — because a bad
    phrasing repeats until it ages out of the history window, so one bad turn
    lands as a streak of failures rather than one. Both the headline and the
    per-class counts therefore need a spread, not just a value.
    """
    rates = [r["clean_rate"] for r in results]
    mean = sum(rates) / len(rates)
    # population stdev; n is small and deliberately so
    sd = (sum((x - mean) ** 2 for x in rates) / len(rates)) ** 0.5

    per_class = {}
    for c in ERROR_CLASSES + WARN_CLASSES + ["collapse"]:
        counts = [r["classes"].get(c, 0) for r in results]
        per_class[c] = {"mean": round(sum(counts) / len(counts), 2),
                        "max": max(counts), "per_run": counts}

    return {
        "runs": len(results),
        "clean_rate_mean": round(mean, 3),
        "clean_rate_sd": round(sd, 3),
        "clean_rate_min": round(min(rates), 3),
        "clean_rate_max": round(max(rates), 3),
        "clean_rate_per_run": [round(x, 3) for x in rates],
        "classes": per_class,
        # the gate's own tolerance, measured rather than guessed: two standard
        # deviations, floored at 5 points so a freakishly tight baseline cannot
        # produce a gate that fires on nothing.
        "tolerance": round(max(0.05, 2 * sd), 3),
    }


def report_aggregate(agg):
    print(f"\npooled over {agg['runs']} run(s)")
    print(f"  clean rate  mean {agg['clean_rate_mean']:.0%}  "
          f"sd {agg['clean_rate_sd']:.1%}  "
          f"range {agg['clean_rate_min']:.0%}-{agg['clean_rate_max']:.0%}  "
          f"({', '.join(f'{x:.0%}' for x in agg['clean_rate_per_run'])})")
    print(f"  gate tolerance {agg['tolerance']:.0%} "
          f"(2 sd, floored at 5 points)")
    rows = [(c, v) for c, v in agg["classes"].items() if v["max"]]
    if rows:
        print("\n  class               mean   max   per run")
        for c, v in sorted(rows, key=lambda kv: -kv[1]["mean"]):
            tag = "warn " if c in WARN_CLASSES else "     "
            print(f"  {tag}{c:<16}{v['mean']:>5}{v['max']:>6}   {v['per_run']}")


def compare(agg, baseline):
    """Gate the pooled result against a pooled baseline.

    The old gate compared one run to one run and failed on a 5 point drop or
    any per-class increase at all. Both fire on chance: the run-to-run range
    was 10 points, and person_mismatch moved 0 -> 9 -> 0 with no code change
    between. The tolerance is now the baseline's own measured spread, and a
    class regresses only when its mean clears what the baseline ever reached.
    """
    print("\nagainst baseline")
    tol = baseline.get("tolerance", 0.05)
    was, now = baseline["clean_rate_mean"], agg["clean_rate_mean"]
    d = now - was
    verdict = "within tolerance" if d >= -tol else "BEYOND TOLERANCE"
    print(f"  clean rate {was:.0%} -> {now:.0%} ({d:+.0%}), "
          f"tolerance -{tol:.0%}  [{verdict}]")
    regressed = d < -tol

    for c in ERROR_CLASSES:
        b = baseline["classes"].get(c, {"mean": 0, "max": 0})
        a = agg["classes"].get(c, {"mean": 0, "max": 0})
        # A class regresses when its average clears the worst the baseline ever
        # saw. One bad run inside the established range is not a regression.
        bad = a["mean"] > max(b["max"], b["mean"] + 1)
        if a["mean"] != b["mean"] or bad:
            flag = "  <-- REGRESSION" if bad else ""
            print(f"  {c:<18} mean {b['mean']} -> {a['mean']} "
                  f"(baseline max {b['max']}){flag}")
        if bad:
            regressed = True

    if regressed:
        print("\nREGRESSION against baseline")
    else:
        print("\nno regression")
    return regressed


# ---------------------------------------------------------------- entry point

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(HERE / "questions.yaml"))
    ap.add_argument("--turns", action="append", metavar="JSONL",
                    help="turns.jsonl from a run; repeat it to pool several "
                         "runs, which is what the gate needs")
    ap.add_argument("--list", action="store_true", help="print the 50 prompts and exit")
    ap.add_argument("--json", help="write the full result as JSON")
    ap.add_argument("--baseline", help="compare against a saved baseline")
    ap.add_argument("--save-baseline", help="write these runs as the baseline")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    questions = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))["questions"]

    if args.list:
        try:
            for q in questions:
                print(f"{q['id']}\t{q['text']}")
        except BrokenPipeError:  # piping into head and friends
            pass
        return 0

    if not args.turns:
        ap.error("--turns is required unless --list is given")

    results = [score(questions, load_turns(t)) for t in args.turns]
    for path, res in zip(args.turns, results):
        if len(results) > 1:
            print(f"\n=== {path} ===")
        report(res, verbose=args.verbose)

    agg = aggregate(results)
    if len(results) > 1:
        report_aggregate(agg)

    if args.json:
        Path(args.json).write_text(
            json.dumps({"runs": results, "aggregate": agg},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_baseline:
        if len(results) < 3:
            print(f"\nrefusing to write a baseline from {len(results)} run(s): "
                  f"the tolerance is derived from the spread, and fewer than "
                  f"three runs cannot show one. Pass --turns repeatedly.")
            return 2
        agg["sources"] = list(args.turns)
        Path(args.save_baseline).write_text(
            json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nbaseline written to {args.save_baseline} "
              f"({agg['runs']} runs, tolerance {agg['tolerance']:.0%})")

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        if compare(agg, base):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
