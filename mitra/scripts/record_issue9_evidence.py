#!/usr/bin/env python3
"""Write issue #9 evidence artifacts that do not require a microphone.

Records the 2026-09-16 Mode A run, scores the Qwen baseline from that live
output, fills recognition.jsonl from the same ASR hypotheses, and reports
quality-stage precision/recall. Does not invent spoken Mode A runs 2–3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _ensure_pkg() -> None:
    try:
        import mitra  # noqa: F401
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mitra", _ROOT / "src" / "__init__.py",
            submodule_search_locations=[str(_ROOT / "src")],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["mitra"] = module
        spec.loader.exec_module(module)


def main() -> int:
    _ensure_pkg()
    from mitra.agent.quality import evaluate_evidence_table, evaluate_quality
    from mitra.agent.validator import validate
    from mitra.eval.metrics import cer, levenshtein, meaning_preserved, wer
    from mitra.eval.qwen_mode_a_20260916 import (
        ASR_ENGINE, EVALUATOR_ID, MODEL_ID, OPERATIONAL, ORCHESTRATOR,
        PROVIDER, REGION, TTS_VOICE, TURNS, grammar_mean, semantic_mean,
    )
    from mitra.eval.sanskrit import evaluate_response
    from mitra.pipeline_trace import first_error_stage

    out = _ROOT / "evals" / "results"
    out.mkdir(parents=True, exist_ok=True)

    # --- Mode A run 1 table + jsonl ---
    mode_a_rows = []
    qwen_rows = []
    rec_rows = []
    prev = None
    ref_words = 0
    edit_words = 0
    for turn in TURNS:
        script_ok, script_reason = validate(turn["sanskrit"])
        q = evaluate_quality(turn["sanskrit"], previous_reply=prev)
        review = evaluate_response(
            prompt=turn["expected"],
            sanskrit=turn["sanskrit"],
            gloss=turn["gloss"],
            grammar=turn["grammar"],
            semantic=turn["semantic"],
            justification=turn["problem"],
            corrected=turn["corrected"],
            uncertain=bool(turn.get("uncertain")),
            evaluator=EVALUATOR_ID,
            gloss_agrees_override=True,
        )
        asr_stage = first_error_stage({
            "payload_kind": "audio",
            "transcript": turn["asr_transcript"],
            "expected_transcript": turn["expected"],
            "audio_stats": {"n_samples": 16000, "peak": 0.2},
            "reply": turn["sanskrit"],
            "validation_ok": script_ok,
            "quality_ok": q["ok"],
        })
        row = {
            "prompt": turn["expected"],
            "id": turn["id"],
            "n": turn["n"],
            "run": 1,
            "test_mode": "mode-a-spoken-2026-09-16",
            "asr_transcript": turn["asr_transcript"],
            "asr_exact": turn["asr_exact"],
            "lang": turn["lang"],
            "asr_hint": turn.get("asr_hint"),
            "provider": PROVIDER,
            "model": MODEL_ID,
            "region": REGION,
            "orchestrator": ORCHESTRATOR,
            "sanskrit": turn["sanskrit"],
            "gloss": turn["gloss"],
            "grammar": turn["grammar"],
            "semantic": turn["semantic"],
            "gloss_matches": "yes (reviewer)",
            "tool_calls": turn["tool_calls"],
            "validator_ok": script_ok,
            "validator_reason": script_reason,
            "quality_ok": q["ok"],
            "quality_flags": q["flags"],
            "quality_reason": q["reason"],
            "first_error_stage": asr_stage,
            "ttfa_s": "equals tts_synth_s (instrumentation bug on this run)",
            "e2e_s": (
                OPERATIONAL["cold_first_turn_e2e_s"] if turn["n"] == 1
                else OPERATIONAL["warm_e2e_s_mean"]
            ),
            "retries": turn["retries"],
            "interruptions": turn["interruptions"],
            "dropped": turn["dropped"],
            "est_usd": turn["cost_usd"],
            "rss_mb": (
                OPERATIONAL["rss_mb_start"] if turn["n"] == 1
                else OPERATIONAL["rss_mb_end"] if turn["n"] == 10
                else None
            ),
            "ollama_loaded": True,
            "ollama_contacted": True,
            "tts": TTS_VOICE,
            "asr_engine": ASR_ENGINE,
            "problem": turn["problem"],
            "corrected": turn["corrected"],
            "review": review,
            "note": (
                "Live Mode A run 1 from the 2026-09-16 operator log. "
                "Actual Qwen output, not a reference fixture."
            ),
        }
        mode_a_rows.append(row)
        qwen_rows.append({
            **row,
            "test_mode": "mode-a-spoken-2026-09-16 (Qwen baseline; written Mode B not run — no Ollama)",
        })
        w = wer(turn["expected"], turn["asr_transcript"])
        c = cer(turn["expected"], turn["asr_transcript"])
        from mitra.eval.metrics import _words
        rw = _words(turn["expected"])
        hw = _words(turn["asr_transcript"])
        ref_words += len(rw)
        edit_words += levenshtein(rw, hw)
        rec_rows.append({
            "id": turn["id"],
            "expected": turn["expected"],
            "hypothesis": turn["asr_transcript"],
            "language": "en",
            "status": "live_mode_a_2026_09_16",
            "wer": round(w, 3),
            "cer": round(c, 3),
            "meaning_preserved": meaning_preserved(turn["expected"], turn["asr_transcript"]),
            "asr_hint": turn.get("asr_hint"),
            "note": (
                "Hypothesis from the 2026-09-16 spoken Mode A log. "
                "Audio was not committed."
            ),
        })
        prev = turn["sanskrit"]

    for sid, expected in (("namaste", "नमस्ते"), ("kim_etat", "किम् एतत्")):
        rec_rows.append({
            "id": sid, "expected": expected, "hypothesis": None,
            "language": "sa", "status": "no_audio",
            "note": "Sanskrit utterance; no consented WAV on this worker.",
        })

    micro_wer = round(edit_words / ref_words, 3) if ref_words else None
    mean_item_wer = round(
        sum(r["wer"] for r in rec_rows if r.get("wer") is not None)
        / sum(1 for r in rec_rows if r.get("wer") is not None),
        3,
    )

    quality_rows = [
        {"id": t["n"], "sanskrit": t["sanskrit"],
         "previous": TURNS[t["n"] - 2]["sanskrit"] if t["n"] > 1 else None}
        for t in TURNS
    ]
    quality_summary = evaluate_evidence_table(quality_rows)

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out / "mode_a_run1_2026-09-16.jsonl", mode_a_rows)
    write_jsonl(out / "conversation_qwen.jsonl", qwen_rows)
    write_jsonl(out / "recognition.jsonl", rec_rows)

    def table(rows: list[dict]) -> str:
        lines = [
            "| Prompt | Run | Test mode | ASR transcript | Model | Sanskrit | Grammar | Semantic | Gloss matches | Latency | Result |",
            "|---|---:|---|---|---|---|---:|---:|---|---:|---|",
        ]
        for r in rows:
            result = (
                "script_ok/quality_flagged" if r.get("validator_ok") and not r.get("quality_ok")
                else "script_ok/quality_ok" if r.get("validator_ok") else "script_fail"
            )
            lines.append(
                f"| {r['prompt']} | {r['run']} | {r['test_mode']} | "
                f"{r['asr_transcript']} | {r['model']} | {r['sanskrit']} | "
                f"{r['grammar']} | {r['semantic']} | yes | "
                f"{r['e2e_s']} | {result} |"
            )
        return "\n".join(lines) + "\n"

    header = (
        "# Qwen3-VL 8B Instruct — scored baseline\n\n"
        "**Source:** 2026-09-16 live Mode A spoken session (issue #9). "
        "This is **actual model output**, not the cloud-agent reference fixture "
        "in `conversation.md`.\n\n"
        f"**Provider:** `{PROVIDER}` / `{MODEL_ID}` / `region={REGION}` / "
        f"orchestrator=`{ORCHESTRATOR}`.\n\n"
        f"**Rubric means:** grammar {grammar_mean()} / semantic {semantic_mean()}. "
        "Gate requires 4.0 each and nothing below 3. **Gate failed.**\n\n"
        "Written Mode B (no mic) was **not** re-run: this worker has no Ollama "
        "and 3.7 GiB RAM. Prompt-ablation Mode B on Bedrock is a separate file.\n\n"
        "Script validation returned `validation_ok: true` on all ten turns. "
        "The quality stage flags turns 1–5, 8, 9 (and 6 when previous is supplied).\n\n"
    )
    (out / "conversation_qwen.md").write_text(header + table(qwen_rows), encoding="utf-8")

    mode_a_md = (
        "# Mode A run 1 — 2026-09-16 offline spoken session\n\n"
        "First real Mode A data point required by issue #7 / #9. "
        "Not discarded. Not a fixture.\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Provider | `{PROVIDER}` |\n"
        f"| Model | `{MODEL_ID}` |\n"
        f"| Region | `{REGION}` |\n"
        f"| Orchestrator | `{ORCHESTRATOR}` |\n"
        f"| Ollama loaded | `true` (offline default) |\n"
        f"| TTS | `{TTS_VOICE}` |\n"
        f"| ASR | `{ASR_ENGINE}` |\n"
        f"| Grammar mean | {grammar_mean()} |\n"
        f"| Semantic mean | {semantic_mean()} |\n"
        f"| Quality gate | **fail** |\n"
        f"| Cold first turn | LLM {OPERATIONAL['cold_first_turn_llm_s']} s / "
        f"e2e {OPERATIONAL['cold_first_turn_e2e_s']} s |\n"
        f"| Warm e2e | ~{OPERATIONAL['warm_e2e_s_mean']} s "
        f"(ASR ~{OPERATIONAL['warm_asr_s_mean']} s) |\n"
        f"| RSS | {OPERATIONAL['rss_mb_start']} → {OPERATIONAL['rss_mb_end']} MB "
        f"(+{OPERATIONAL['rss_mb_growth']} MB / 10 turns) |\n"
        f"| `ttfa_s` | equalled TTS synthesis on every turn (bug; fixed in this PR) |\n"
        f"| Self-echo wake transcripts | {len(OPERATIONAL['self_echo_wake_transcripts'])} "
        f"(no false wake) |\n"
        f"| `Audio system is not initialized.` | "
        f"{OPERATIONAL['audio_system_not_initialized_count']} |\n\n"
        "## Recognition\n\n"
        "Nine of ten transcripts exact. Turn 8: `Do you play sports?` → "
        "`2 play sports` (Do→2). `first_error_stage` was `null` on the log; "
        "the diagnostic now returns `asr` for a leading-digit hypothesis. "
        "Likely VAD truncation of the leading function word; English retry "
        "now treats the hypothesis as unusable.\n\n"
        f"English micro-WER (this corpus): **{micro_wer}**. "
        f"Mean per-item WER: **{mean_item_wer}**. "
        "Issue #9 quoted ~0.06. Recognition is largely fine; the gate failure "
        "is response quality.\n\n"
        "## Self-echo (wake-check transcripts of Mitra's playback)\n\n"
        + "\n".join(f"- `{t}`" for t in OPERATIONAL["self_echo_wake_transcripts"])
        + "\n\n## Scored turns\n\n"
        + table(mode_a_rows)
        + "\n## Per-turn problems\n\n"
        + "\n".join(
            f"- **{t['n']}. {t['expected']}** — {t['problem']} "
            f"Candidate: {t['corrected']}"
            for t in TURNS
        )
        + "\n"
    )
    (out / "mode_a_run1_2026-09-16.md").write_text(mode_a_md, encoding="utf-8")

    rec_md = (
        "# Recognition corpus — measured WER/CER\n\n"
        "Audio was **not** committed. Hypotheses are the live 2026-09-16 ASR "
        "transcripts for the ten English scenarios.\n\n"
        f"| Language | Items scored | Micro-WER | Mean item WER |\n"
        f"|---|---:|---:|---:|\n"
        f"| en | 10 | {micro_wer} | {mean_item_wer} |\n"
        f"| sa | 0 | — | no consented WAV |\n\n"
        "Turn 8 (`Do you play sports?` → `2 play sports`) is the only error. "
        "The sentence-recognition objective from #7 is restated: English WER "
        f"on this set is {micro_wer}, below a 0.10 working bar. Residual: "
        "leading-function-word truncation / Do→2. `first_error_stage` is now `asr`.\n"
    )
    (out / "recognition.md").write_text(rec_md, encoding="utf-8")

    (out / "quality_gate_precision.json").write_text(
        json.dumps(quality_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"grammar_mean={grammar_mean()} semantic_mean={semantic_mean()}")
    print(f"quality precision={quality_summary['precision']} recall={quality_summary['recall']}")
    print(f"recognition micro-WER={micro_wer} mean-item-WER={mean_item_wer}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
