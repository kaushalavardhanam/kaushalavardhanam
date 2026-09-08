"""Orchestrator state machine (DESIGN §3–§4).

States: ASLEEP → WAKING → LISTENING → THINKING → SPEAKING → LISTENING … → ASLEEP.

Single-threaded core: all transitions happen in ``handle_event`` on the run
loop's thread. Two daemon helpers — the audio pump and the playback watcher —
communicate with the core only by putting events on the queue (DESIGN §3).
Tests drive ``handle_event`` directly with fakes; ``run()`` adds the threads.

The agent may call tools itself, but two paths stay deterministic regardless
of model quality (DESIGN §1.4): ``nod`` fires here on wake, and every reply
passes the validator and is spoken here.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from mitra import language_detector
from mitra.agent import prompts, validator
from mitra.agent.tools import END_SESSION_SENTINEL
from mitra.audio import TARGET_SAMPLERATE, resample


# "explain in English" detection (FR-3.2 exception): explicit request only —
# ordinary English questions still get Sanskrit answers.
_EXPLAIN_IN_ENGLISH_RE = re.compile(
    r"in\s+english|english\s*,?\s*please|(explain|meaning|translate|repeat)"
    r"[\w\s,]{0,30}english", re.IGNORECASE)


class State(str, Enum):
    ASLEEP = "ASLEEP"
    WAKING = "WAKING"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


@dataclass
class Event:
    kind: str  # wake | utterance | playback_done | tick | stop
    payload: object = None


class Orchestrator:
    def __init__(self, *, robot, agent, tts, lexicon,
                 wake=None, segmenter=None, asr=None,
                 turn_logger=None, logger: logging.Logger | None = None,
                 silence_timeout_s: float = 30.0,
                 max_reply_chars: int = validator.MAX_REPLY_CHARS,
                 fallback_agent_factory=None, gestures: bool = True,
                 llm_meta: dict | None = None):
        self.robot = robot
        self.agent = agent
        self.tts = tts
        self.lexicon = lexicon
        self.wake = wake
        self.segmenter = segmenter
        self.asr = asr
        self.turn_logger = turn_logger
        self.logger = logger or logging.getLogger("mitra")
        self.gestures = gestures
        self.silence_timeout_s = silence_timeout_s
        self.max_reply_chars = max_reply_chars
        self._fallback_agent_factory = fallback_agent_factory
        self._fallback_agent = None
        self.llm_meta = dict(llm_meta or {})
        if not self.llm_meta:
            self.llm_meta = {
                "provider": getattr(agent, "provider", None),
                "model_id": getattr(agent, "model_id", None),
                "region": getattr(agent, "region", None),
            }

        self.state = State.ASLEEP
        self.events: queue.Queue[Event] = queue.Queue()
        self._stop = threading.Event()
        self._sleep_after_speaking = False
        self._last_activity = time.monotonic()

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        self._stop.clear()
        threading.Thread(target=self._audio_loop, daemon=True).start()
        self.logger.info("Mitra asleep — say the wake word")
        while not self._stop.is_set():
            try:
                event = self.events.get(timeout=0.5)
            except queue.Empty:
                event = Event("tick")
            try:
                self.handle_event(event)
            except Exception:  # FR-6.4: log, apologize, keep the session alive
                self.logger.exception("error handling %s in %s", event.kind, self.state)
                if self.state == State.THINKING:
                    self._emotion("confused1")
                    self._speak(prompts.APOLOGY_RETRY)
                    self.state = State.SPEAKING

    def stop(self) -> None:
        self.events.put(Event("stop"))

    # ------------------------------------------------------- event dispatch

    def handle_event(self, event: Event) -> None:
        kind = event.kind
        if kind == "stop":
            self._stop.set()
        elif kind == "tick":
            self._check_silence_timeout()
        elif kind == "wake":
            if self.state == State.ASLEEP:
                self._on_wake()
            elif self.state in (State.SPEAKING, State.WAKING):
                # barge-in (DESIGN §1.3): stop playback, listen again
                self.robot.speaker_stop()
                self._to_listening()
        elif kind == "utterance" and self.state == State.LISTENING:
            self._on_utterance(event.payload)
        elif kind == "playback_done":
            if self._sleep_after_speaking:
                self._go_to_sleep()
            elif self.state in (State.WAKING, State.SPEAKING):
                self._to_listening()

    # ---------------------------------------------------------- transitions

    def _pose(self, name: str) -> None:
        """State-feedback gesture (FR-5.3): best-effort, config-gated."""
        if self.gestures and hasattr(self.robot, "pose"):
            self.robot.pose(name)

    def _emotion(self, name: str) -> None:
        """Recorded emotion from Pollen's library (FR-5.3 extension):
        best-effort, config-gated, purely additive to _pose() above — fired
        only at occasional "moments" (wake, sleep, confusion), not every turn."""
        if self.gestures and hasattr(self.robot, "play_emotion"):
            self.robot.play_emotion(name)

    def _on_wake(self) -> None:
        self.state = State.WAKING
        self.logger.info("wake word detected")
        self.robot.nod()                      # deterministic (DESIGN §1.4)
        self._emotion("welcoming1")
        self._speak(prompts.GREETING)         # → playback_done → LISTENING

    def _to_listening(self) -> None:
        self.state = State.LISTENING
        self._last_activity = time.monotonic()
        self._pose("listening")               # antennas perk up: "your turn"
        if self.segmenter:
            self.segmenter.reset()

    def _check_silence_timeout(self) -> None:
        if (self.state == State.LISTENING
                and time.monotonic() - self._last_activity > self.silence_timeout_s):
            self.logger.info("silence timeout — session ends (FR-1.5)")
            self._go_to_sleep()

    def _go_to_sleep(self) -> None:
        self.state = State.ASLEEP
        self._pose("asleep")                  # head droops: session over
        self._emotion("mini-deep-sleep")
        self._sleep_after_speaking = False
        self.agent.reset()                    # context is per-session (FR-3.3)
        if self.wake:
            self.wake.reset()
        if self.segmenter:
            self.segmenter.reset()
        self.logger.info("asleep")

    def _on_utterance(self, payload) -> None:
        self.state = State.THINKING
        self._pose("thinking")                # head tilt: "processing..."
        self._last_activity = time.monotonic()
        turn_t0 = time.monotonic()
        tl = self.turn_logger
        if tl:
            tl.start_turn()
            self._bind_turn_identity(tl, payload)

        transcript, hint = self._transcribe(payload)
        if tl:
            self._bind_asr_diag(tl)
        if not transcript.strip():
            if tl:
                tl.set("llm_empty", False)
                tl.set("asr_empty", True)
            self._finish_turn(prompts.APOLOGY_RETRY, turn_t0=turn_t0)
            return

        lang = language_detector.detect(transcript, hint)
        explain_en = bool(_EXPLAIN_IN_ENGLISH_RE.search(transcript))
        message = (f"[lang={lang}] [explain_in_english] {transcript}"
                   if explain_en else f"[lang={lang}] {transcript}")
        if tl:
            tl.set("lang", lang)
            tl.set("transcript", transcript)
            tl.set("asr_hint", hint)
            tl.set("explain_in_english", explain_en)

        try:
            reply, session_end = self._generate_reply(message, explain_en)
        except Exception:
            self.logger.exception("agent failure (FR-6.4)")
            if tl:
                tl.set("llm_error", True)
            self._emotion("confused1")
            reply, session_end = prompts.APOLOGY_RETRY, False

        if session_end:
            self._sleep_after_speaking = True
            reply = prompts.FAREWELL
        self._finish_turn(reply, turn_t0=turn_t0)

    def _bind_turn_identity(self, tl, payload) -> None:
        tl.set("provider", self.llm_meta.get("provider"))
        tl.set("model_id", self.llm_meta.get("model_id"))
        tl.set("region", self.llm_meta.get("region"))
        if isinstance(payload, str):
            tl.set("payload_kind", "text")
            return
        tl.set("payload_kind", "audio")
        from mitra.pipeline_trace import audio_stats

        tl.set("audio_stats", audio_stats(payload, TARGET_SAMPLERATE))

    def _bind_asr_diag(self, tl) -> None:
        asr = self.asr
        if asr is None or not getattr(asr, "last_diag", None):
            return
        diag = asr.last_diag
        tl.set("asr_hallucination", bool(diag.get("hallucination")))
        if diag.get("transcript") is not None:
            tl.set("asr_raw", diag.get("transcript"))
        if diag.get("audio_stats"):
            tl.set("audio_stats", diag["audio_stats"])
        tl.set("asr_english_retry", bool(diag.get("english_retry")))
        tl.set("asr_low_energy", bool(diag.get("low_energy")))

    def _finish_turn(self, reply: str, turn_t0: float | None = None) -> None:
        tl = self.turn_logger
        if tl:
            tl.set("reply", reply)
        self._pose("neutral")                 # face forward while speaking
        tts_t0 = time.monotonic()
        if tl:
            with tl.stage("tts"):
                self._speak(reply)
            tl.set("ttfa_s", round(time.monotonic() - tts_t0, 3))
            if turn_t0 is not None:
                tl.set("e2e_s", round(time.monotonic() - turn_t0, 3))
            tl.emit()
        else:
            self._speak(reply)
        self.state = State.SPEAKING

    def _transcribe(self, payload) -> tuple[str, str | None]:
        if isinstance(payload, str):          # tests / text console mode
            return payload, None
        tl = self.turn_logger
        if tl:
            with tl.stage("asr"):
                return self.asr.transcribe(payload)
        return self.asr.transcribe(payload)

    # ------------------------------------------------------------ thinking

    def _generate_reply(self, message: str,
                        explain_en: bool = False) -> tuple[str, bool]:
        """Agent call + lexicon substitution + validation with one retry
        (FR-3.5), then the config-gated cloud fallback (FR-6.3).

        When the user explicitly asked for an English explanation (FR-3.2
        exception), the Devanagari check is waived for this one turn — the
        reply must merely be non-empty and not a ramble."""
        tl = self.turn_logger

        def generate(msg: str) -> str:
            if tl:
                with tl.stage("llm"):
                    return self.agent.converse(msg)
            return self.agent.converse(msg)

        raw = generate(message)
        if END_SESSION_SENTINEL in raw:
            if tl:
                tl.set("tool_end_session", True)
            return raw, True

        if explain_en:
            reply = raw.strip()
            if reply and len(reply) <= 3 * self.max_reply_chars:
                if tl:
                    tl.set("validation_ok", True)
                    tl.set("validation_reason", "explain_in_english")
                return reply, False
            if tl:
                tl.set("validation_ok", False)
                tl.set("validation_reason", "english_explanation_unusable")
            return prompts.SAFE_FALLBACK, False

        reply = self._apply_lexicon(raw)
        ok, reason = validator.validate(reply, self.max_reply_chars)
        if tl:
            tl.set("raw_reply", raw)
            tl.set("validation_ok", ok)
            tl.set("validation_reason", reason)
            tl.set("lexicon_applied", reply != raw)
        if ok:
            return reply, False

        self.logger.warning("reply failed validation (%s); retrying", reason)
        if tl:
            tl.set("validation_retry", True)
        reply = self._apply_lexicon(
            generate(message + "\n" + prompts.CORRECTIVE_SUFFIX)
        )
        ok, reason = validator.validate(reply, self.max_reply_chars)
        if tl:
            tl.set("validation_ok", ok)
            tl.set("validation_reason", reason)
        if ok:
            return reply, False

        self.logger.warning("retry failed validation (%s)", reason)
        cloud = self._try_cloud_fallback(message)
        return (cloud if cloud is not None else prompts.SAFE_FALLBACK), False

    def _try_cloud_fallback(self, message: str) -> str | None:
        if self._fallback_agent_factory is None:
            return None
        try:
            if self._fallback_agent is None:
                self._fallback_agent = self._fallback_agent_factory()
            reply = self._apply_lexicon(self._fallback_agent.converse(message))
            ok, _ = validator.validate(reply, self.max_reply_chars)
            return reply if ok else None
        except Exception:
            self.logger.exception("cloud fallback failed (FR-6.3)")
            return None

    # ------------------------------------------------------- vision/lexicon

    def _apply_lexicon(self, reply: str) -> str:
        """Vision turns answer in strict JSON (DESIGN §5). Verified lexicon
        names always override the generated name (FR-2.5); new names are
        recorded unverified for review (DESIGN §4)."""
        data = _extract_json(reply)
        if not data or "object_en" not in data:
            return reply
        object_en = str(data["object_en"])
        generated = str(data.get("name_sa_devanagari", "")).strip()
        sentence = str(data.get("sentence_sa", "")).strip()
        if not sentence and generated:
            sentence = f"एतत् {generated} अस्ति।"

        row = self.lexicon.lookup(object_en)
        if row and row["verified"]:
            verified_name = row["name_devanagari"]
            if generated and generated in sentence:
                sentence = sentence.replace(generated, verified_name)
            else:
                sentence = f"एतत् {verified_name} अस्ति।"
        elif row is None and generated:
            self.lexicon.add_unverified(
                object_en, generated, str(data.get("name_iast", "")), object_en
            )
        return sentence or reply

    # ------------------------------------------------------------- speaking

    def _speak(self, text: str) -> None:
        """Deterministic speech path (DESIGN §1.4): synthesize, play without
        blocking (for barge-in), post playback_done when the speaker frees up."""
        self.logger.info("speak: %s", text)
        try:
            wav, samplerate = self.tts.synthesize(text)
            self.robot.speaker_play(wav, samplerate, block=False)
        except Exception:
            self.logger.exception("TTS/playback failure (FR-6.4)")
            self.events.put(Event("playback_done"))
            return
        threading.Thread(target=self._watch_playback, daemon=True).start()

    def _watch_playback(self) -> None:
        time.sleep(0.05)
        while self.robot.speaker_busy() and not self._stop.is_set():
            time.sleep(0.05)
        # Discard whatever the mic captured during playback before resuming
        # listening — with mic_source="built_in" there's no echo cancellation,
        # so without this the robot's own voice gets fed back in as if it
        # were the next user utterance.
        if hasattr(self.robot, "flush_mic"):
            self.robot.flush_mic()
        self.events.put(Event("playback_done"))

    # ------------------------------------------------------------ audio I/O

    def _audio_loop(self) -> None:
        """Pump mic chunks to the wake detector or segmenter by state."""
        while not self._stop.is_set():
            try:
                chunk = self.robot.mic_read()
            except Exception:
                self.logger.exception("microphone read failure")
                time.sleep(0.5)
                continue
            if chunk is None or len(chunk) == 0:
                continue
            if self.robot.mic_samplerate != TARGET_SAMPLERATE:
                chunk = resample(chunk, self.robot.mic_samplerate, TARGET_SAMPLERATE)

            state = self.state
            if state in (State.ASLEEP, State.SPEAKING, State.WAKING):
                if self.wake and self.wake.process(chunk):
                    self.events.put(Event("wake"))
            elif state == State.LISTENING and self.segmenter is not None:
                utterance = self.segmenter.process(chunk)
                if utterance is not None:
                    self.events.put(Event("utterance", utterance))


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a reply (tolerates ``` fences)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
