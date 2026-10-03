"""GPT5.6 client for the MITRA coherence check.

Three jobs, all going through one cached, JSON-structured call path:

  1. followup_reply(history)   Sanskrit reply that continues a dialogue after
                               the first exchange.
  2. baseline_reply(history)   Plain GPT5.6 reply to the same prompt(s), for
                               comparison with the Reachy Mini agent.
  3. score_metric(metric, ...) Score a metric that cannot be computed
                               programmatically, using a fixed rubric.

Configuration comes from environment variables (see mitra/.env.example):
    OPENAI_API_KEY      required
    MITRA_JUDGE_MODEL   required, the GPT5.6 identifier from your account
    OPENAI_BASE_URL     optional, default https://api.openai.com/v1
    MITRA_JUDGE_CACHE   optional, cache directory
                        (default mitra/logs/judge_cache)

History format: a list of {"role": "user" | "assistant", "text": "..."}.
The role "agent" is accepted as an alias for "assistant". The returned reply
is the next "assistant" turn.

Caching: every request is keyed by a hash of (kind, model, temperature,
messages, sample). Identical requests are served from disk, so re-running the
harness costs nothing. Pass a different ``sample`` (e.g. the run number) to get
an independent generation for the same prompt when temperature > 0.

Some models only accept the default temperature. If the API rejects the
temperature parameter the client retries once without it and remembers that.

Usage (from the mitra/ directory):

    from coherence.judge_client import JudgeClient
    client = JudgeClient.from_env()
    reply = client.followup_reply(history, sample=run)
    base = client.baseline_reply(history, sample=run)
    result = client.score_metric("coherence", reply, history=history)

CLI smoke tests:
    python -m coherence.judge_client baseline "कथं भवान्?"
    python -m coherence.judge_client score coherence "text to score"
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "logs" / "judge_cache"

REPLY_TEMPERATURE = 0.7  # dialogue generation: some variety between runs
SCORE_TEMPERATURE = 0.0  # scoring: as deterministic as possible

SCORE_MIN = 1
SCORE_MAX = 5

REPLY_SYSTEM_PROMPT = (
    "You are a friendly conversational partner who speaks only Sanskrit, "
    "written in Devanagari script. Reply in short, grammatically correct, "
    "simple spoken Sanskrit (one to three sentences), suitable for a spoken "
    "dialogue. Do not use Hindi, English or transliteration. Stay on the "
    "topic of the conversation and answer what was asked. "
    'Respond ONLY with a JSON object of the form {"reply": "<Sanskrit text>"}.'
)

SCORE_SYSTEM_PROMPT = (
    "You are a strict, consistent evaluator of spoken Sanskrit dialogue. "
    f"Score on an integer scale from {SCORE_MIN} (very poor) to {SCORE_MAX} "
    "(excellent) using ONLY the rubric given for the metric. Do not reward "
    "length. Judge the text itself, not the effort involved. "
    'Respond ONLY with a JSON object: {"score": <integer>, '
    '"rationale": "<one or two sentences>"}.'
)

# Rubrics: each metric describes what 1, 3 and 5 mean. Add metrics here.
RUBRICS = {
    "coherence": (
        "Does the reply logically follow the preceding dialogue without "
        "contradicting it or losing the thread? "
        "1 = unrelated or contradicts earlier turns; "
        "3 = loosely related but vague or partly off-topic; "
        "5 = clearly continues the dialogue and is consistent with it."
    ),
    "relevance": (
        "Does the reply address what the last speaker actually said or asked? "
        "1 = ignores it; 3 = addresses it only partly; "
        "5 = fully and directly addresses it."
    ),
    "fluency": (
        "Is the text natural, idiomatic Sanskrit for spoken dialogue? "
        "1 = broken or unintelligible; 3 = understandable but awkward; "
        "5 = natural and idiomatic."
    ),
    "language_purity": (
        "Is the text entirely Sanskrit, with no Hindi, English or other "
        "language mixed in? 1 = mostly another language; "
        "3 = some Hindi/English words or constructions; "
        "5 = entirely Sanskrit."
    ),
}


class JudgeError(RuntimeError):
    """Raised when the judge model cannot be reached or returns unusable output."""


def _normalize_history(history):
    messages = []
    for turn in history or []:
        role = turn.get("role", "user")
        role = "assistant" if role in ("assistant", "agent") else "user"
        text = turn.get("text", turn.get("content", ""))
        messages.append({"role": role, "content": text})
    return messages


def _extract_json(raw):
    """Parse a JSON object from model output, tolerating code fences and prose."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


class JudgeClient:
    def __init__(self, api_key, model, base_url=None, cache_dir=None,
                 use_cache=True, timeout=60, max_retries=4):
        if not api_key:
            raise JudgeError("OPENAI_API_KEY is not set")
        if not model:
            raise JudgeError("MITRA_JUDGE_MODEL is not set")
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.use_cache = use_cache
        self.timeout = timeout
        self.max_retries = max_retries
        self._send_temperature = True
        self.api_calls = 0
        self.cache_hits = 0

    @classmethod
    def from_env(cls, **kwargs):
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("MITRA_JUDGE_MODEL"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            cache_dir=os.environ.get("MITRA_JUDGE_CACHE"),
            **kwargs,
        )

    # ---- public API -------------------------------------------------------

    def followup_reply(self, history, temperature=REPLY_TEMPERATURE, sample=0):
        """Sanskrit reply continuing the dialogue in ``history``."""
        return self._reply("followup", history, temperature, sample)

    def baseline_reply(self, history, temperature=REPLY_TEMPERATURE, sample=0):
        """Baseline GPT5.6 reply to the same prompt(s) the agent received.

        Uses the same system prompt and settings as followup_reply so the two
        differ only in the conversation given; the ``kind`` label keeps their
        cache entries separate.
        """
        return self._reply("baseline", history, temperature, sample)

    def score_metric(self, metric, text, history=None, temperature=SCORE_TEMPERATURE,
                     sample=0):
        """Score ``text`` for ``metric``; returns {"score": int, "rationale": str}."""
        rubric = RUBRICS.get(metric)
        if rubric is None:
            raise JudgeError(f"Unknown metric {metric!r}; known: {sorted(RUBRICS)}")
        payload = {
            "metric": metric,
            "rubric": rubric,
            "scale": {"min": SCORE_MIN, "max": SCORE_MAX},
            "dialogue_so_far": [
                {"role": m["role"], "text": m["content"]}
                for m in _normalize_history(history)
            ],
            "text_to_score": text,
        }
        messages = [
            {"role": "system", "content": SCORE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        data = self._structured(f"score:{metric}", messages, temperature, sample)
        try:
            score = int(round(float(data["score"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise JudgeError(f"Score missing or invalid in {data!r}") from exc
        score = max(SCORE_MIN, min(SCORE_MAX, score))
        return {"score": score, "rationale": str(data.get("rationale", ""))}

    # ---- internals --------------------------------------------------------

    def _reply(self, kind, history, temperature, sample):
        messages = [{"role": "system", "content": REPLY_SYSTEM_PROMPT}]
        messages += _normalize_history(history)
        if len(messages) == 1:
            raise JudgeError("history is empty; nothing to reply to")
        if messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "कृपया संवादं प्रवर्तयतु।"})
        data = self._structured(kind, messages, temperature, sample)
        reply = data.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            raise JudgeError(f"No reply in model output: {data!r}")
        return reply.strip()

    def _cache_path(self, kind, messages, temperature, sample):
        key = json.dumps(
            [kind, self.model, temperature, sample, messages],
            ensure_ascii=False, sort_keys=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _structured(self, kind, messages, temperature, sample):
        path = self._cache_path(kind, messages, temperature, sample)
        if self.use_cache and path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)["response"]
                self.cache_hits += 1
                return data
            except (OSError, ValueError, KeyError):
                pass  # corrupt cache entry: fall through and regenerate

        last_error = None
        for _ in range(2):  # one extra try if the output is not valid JSON
            raw = self._chat(messages, temperature)
            try:
                data = _extract_json(raw)
                if isinstance(data, dict):
                    break
                last_error = ValueError("JSON output is not an object")
            except ValueError as exc:
                last_error = exc
        else:
            raise JudgeError(f"Model returned unparseable JSON: {last_error}")

        if self.use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(
                    {"kind": kind, "model": self.model, "temperature": temperature,
                     "sample": sample, "messages": messages, "response": data},
                    f, ensure_ascii=False, indent=2,
                )
        return data

    def _chat(self, messages, temperature):
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if self._send_temperature and temperature is not None:
            body["temperature"] = temperature
        return self._post(body, temperature)

    def _post(self, body, temperature):
        url = f"{self.base_url}/chat/completions"
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                self.api_calls += 1
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 400 and "temperature" in detail and "temperature" in body:
                    # Model only supports its default temperature.
                    self._send_temperature = False
                    body = {k: v for k, v in body.items() if k != "temperature"}
                    continue
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise JudgeError(f"HTTP {exc.code} from judge model: {detail[:500]}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise JudgeError(f"Could not reach judge model: {exc}") from exc
            except (KeyError, IndexError, ValueError) as exc:
                raise JudgeError(f"Unexpected API response shape: {exc}") from exc
        raise JudgeError("Judge model request failed after retries")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GPT5.6 judge client smoke test")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline", help="baseline reply to a single prompt")
    b.add_argument("prompt")
    s = sub.add_parser("score", help="score a text for a metric")
    s.add_argument("metric", choices=sorted(RUBRICS))
    s.add_argument("text")
    args = parser.parse_args(argv)

    try:
        client = JudgeClient.from_env()
        if args.cmd == "baseline":
            print(client.baseline_reply([{"role": "user", "text": args.prompt}]))
        else:
            print(json.dumps(client.score_metric(args.metric, args.text),
                             ensure_ascii=False, indent=2))
    except JudgeError as exc:
        print("Error:", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())