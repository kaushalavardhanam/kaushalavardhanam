"""Orchestrator state machine (DESIGN §3–§4).

States: ASLEEP → WAKING → LISTENING → THINKING → SPEAKING → LISTENING … → ASLEEP.

Single-threaded core: all transitions happen in ``handle_event`` on the run
loop's thread. Two daemon helpers — the audio pump and the playback watcher —
communicate with the core only by putting events on the queue (DESIGN §3).
Tests drive ``handle_event`` directly with fakes; ``run()`` adds the threads.

The agent may call tools itself, but three paths stay deterministic regardless
of model quality (DESIGN §1.4): ``nod`` fires here on wake, unintelligible
transcripts are refused here before the model ever sees them, and every reply
passes the validator and is spoken here.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from contextlib import contextmanager
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

# Whisper emits these on silence or near-silence regardless of what was said —
# they are training-data artifacts (YouTube captions), not transcriptions.
# Reaching the model with one produces a confident answer to a question the
# user never asked, which is worse than asking them to repeat.
_ASR_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks for watching", "thank you very much",
    "thank you for watching", "thank you for watching!", "you", "bye",
    "please subscribe", "subtitles by the amara.org community",
}

# Below this many characters a transcript carries too little signal to answer.
_MIN_TRANSCRIPT_CHARS = 3

# Sentence terminators, Devanagari and Latin. Used to hold the reply to
# ``max_sentences`` (DESIGN §1.4: the model is asked for one sentence, and this
# enforces it whether or not it complies).
_SENTENCE_END = re.compile(r"[।॥?!]")

# Wrapper around retrieved phrasing. The wording is load-bearing: labelled only
# as "reference phrasing", an 8B model pastes the nearest line back verbatim —
# we watched it answer "Sarvam kushalam" with "सर्वं कुशलम्।", which is the
# corpus row, not a reply. Naming what to do beats naming what not to do.
#
# Retrieval now resolves a question to its ANSWER rows, so what arrives here is
# reply-shaped and correctly inflected — the one thing an 8B model is worst at
# inventing. The instruction therefore says adapt, not avoid: copying the
# grammar is the point, copying the content is not. Rows carry "fill" where the
# book left a blank, which must never be spoken.
_REFERENCE_HEADER = (
    "\n[Example Sanskrit replies of the right register. Follow their grammar "
    "and style, but write your OWN sentence about what the user actually said "
    "— never repeat one word-for-word, and never say the word \"fill\"]\n"
)


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
                 wake=None, segmenter=None, asr=None, phrasebook=None,
                 turn_logger=None, glosser=None, grammar_checker=None,
                 logger: logging.Logger | None = None,
                 silence_timeout_s: float = 30.0,
                 max_reply_chars: int = validator.MAX_REPLY_CHARS,
                 max_sentences: int = 1,
                 fallback_agent_factory=None, gestures: bool = True):
        self.robot = robot
        self.agent = agent
        self.tts = tts
        self.lexicon = lexicon
        self.wake = wake
        self.segmenter = segmenter
        self.asr = asr
        # Optional: retrieval over the everyday-phrase corpus. Without it the
        # model has only the few-shot examples to fall back on, and an 8B with
        # weak Sanskrit priors answers an unclear turn by reciting one of them
        # verbatim — which is what makes three exchanges read as one.
        self.phrasebook = phrasebook
        self.turn_logger = turn_logger
        # Optional (debug runs only): translates every spoken line back into
        # English for the log, so an operator who does not read Devanagari can
        # see what Mitra actually said (FR-7.2).
        self.glosser = glosser
        # Optional: morphology-backed checks over the generated reply
        # (mitra.sanskrit.grammar). The Devanagari ratio in validator.py is
        # blind to Hindi written in Devanagari and to invented verb forms;
        # this is what sees them. Absent, the pipeline behaves exactly as it
        # did before — the checks add rejections, they never add replies.
        self.grammar_checker = grammar_checker
        self.logger = logger or logging.getLogger("mitra")
        self.gestures = gestures
        self.silence_timeout_s = silence_timeout_s
        self.max_reply_chars = max_reply_chars
        # Every logged reply appended a stock "and you?" — and that trailing
        # question, not the answer, carried most of the grammatical errors
        # (कथं भवतः?, genitive where nominative is needed). The prompt asks for
        # one sentence; this is what makes it true. Set to 2 to allow a
        # follow-up question back.
        self.max_sentences = max_sentences
        self._fallback_agent_factory = fallback_agent_factory
        self._fallback_agent = None

        self.state = State.ASLEEP
        self.events: queue.Queue[Event] = queue.Queue()
        self._stop = threading.Event()
        self._sleep_after_speaking = False
        self._last_activity = time.monotonic()
        self._consecutive_retries = 0

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
        self._consecutive_retries = 0
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
        self._consecutive_retries = 0
        self.agent.reset()                    # context is per-session (FR-3.3)
        if self.wake:
            self.wake.reset()
        if self.segmenter:
            self.segmenter.reset()
        self.logger.info("asleep")

    # ------------------------------------------------------ turn processing

    def _is_unintelligible(self, transcript: str, lang: str) -> str | None:
        """Return a reason string if this transcript must not reach the model.

        Whisper always returns *something*: on silence it emits a stock caption
        phrase, and on speech in a language it wasn't expecting it transliterates
        into whatever script it guessed (we have seen Korean, Japanese, Arabic
        and Tamil renderings of spoken Sanskrit). Both look like valid input to
        the agent, which then answers confidently — or, worse, echoes the noise
        back as if it were Sanskrit. Refusing here is the only place the
        pipeline can tell the difference.
        """
        cleaned = transcript.strip()
        if len(cleaned) < _MIN_TRANSCRIPT_CHARS:
            return "too short"
        if cleaned.lower().strip(" .!?") in _ASR_HALLUCINATIONS:
            return "known ASR hallucination"
        if lang == "unknown":
            # Script matched none of en/kn/sa: the words may well be right but
            # they are in an alphabet the model has no reason to connect to
            # this conversation.
            return "unrecognized script"
        return None

    def _retrieve_examples(self, transcript: str) -> list:
        """Nearest everyday phrases for this turn, or [] if unavailable."""
        if self.phrasebook is None:
            return []
        try:
            return self.phrasebook.similar(transcript, k=3)
        except Exception:
            self.logger.exception("phrasebook lookup failed; continuing ungrounded")
            return []

    def _build_message(self, transcript: str, lang: str, explain_en: bool) -> str:
        """Assemble the turn message: tags, transcript, retrieved phrasing.

        The examples go after the transcript so the user's turn stays the most
        recent thing in the message — context that follows a question tends to
        get answered instead of the question.
        """
        header = f"[lang={lang}]"
        if explain_en:
            header += " [explain_in_english]"
        message = f"{header} {transcript}"

        examples = self._retrieve_examples(transcript)
        if examples:
            lines = "\n".join(
                f"  {row.get('english', '')} → {row.get('sanskrit', '')}"
                for row in examples
            )
            message += _REFERENCE_HEADER + lines
        return message

    def _on_utterance(self, payload) -> None:
        self.state = State.THINKING
        self._pose("thinking")                # head tilt: "processing..."
        self._last_activity = time.monotonic()
        tl = self.turn_logger
        if tl:
            tl.start_turn()

        transcript, hint = self._transcribe(payload)
        lang = language_detector.detect(transcript, hint) if transcript.strip() else "unknown"

        if tl:
            tl.set("lang", lang)
            tl.set("transcript", transcript)

        reason = self._is_unintelligible(transcript, lang)
        if reason:
            # Do NOT call the model. "Sorry, say that again" is a coherent
            # conversational move; answering noise is not.
            self._consecutive_retries += 1
            self.logger.info("transcript rejected (%s): %r", reason, transcript)
            if tl:
                tl.set("rejected", reason)
                tl.set("explain_in_english", False)
            self._emotion("confused1")
            self._finish_turn(prompts.APOLOGY_RETRY)
            return

        self._consecutive_retries = 0
        explain_en = bool(_EXPLAIN_IN_ENGLISH_RE.search(transcript))
        message = self._build_message(transcript, lang, explain_en)
        if tl:
            tl.set("explain_in_english", explain_en)

        try:
            reply, session_end = self._generate_reply(message, explain_en)
        except Exception:
            self.logger.exception("agent failure (FR-6.4)")
            self._emotion("confused1")
            reply, session_end = prompts.APOLOGY_RETRY, False

        if session_end:
            self._sleep_after_speaking = True
            reply = prompts.FAREWELL
        self._finish_turn(reply)

    def _finish_turn(self, reply: str) -> None:
        tl = self.turn_logger
        if tl:
            tl.set("reply", reply)
        self._pose("neutral")                 # face forward while speaking
        english = self._speak(reply)          # times the tts stage itself
        if tl:
            if english:
                tl.set("reply_en", english)
            tl.emit()
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
            return raw, True

        if explain_en:
            reply = raw.strip()
            if reply and len(reply) <= 3 * self.max_reply_chars:
                return reply, False
            return prompts.SAFE_FALLBACK, False

        reply = _limit_sentences(self._apply_lexicon(raw), self.max_sentences)
        ok, reason, suffix = self._check(reply)
        if ok:
            return reply, False

        self.logger.warning("reply failed validation (%s); retrying", reason)
        reply = _limit_sentences(
            self._apply_lexicon(generate(message + "\n" + suffix)),
            self.max_sentences,
        )
        ok, reason, _ = self._check(reply)
        if ok:
            return reply, False

        self.logger.warning("retry failed validation (%s)", reason)
        cloud = self._try_cloud_fallback(message)
        return (cloud if cloud is not None else prompts.SAFE_FALLBACK), False

    def _check(self, reply: str) -> tuple[bool, str, str]:
        """Validate a candidate reply. Returns (ok, reason, retry suffix).

        Two layers, deliberately separate: ``validator`` is the cheap, always-on
        gate (length, Devanagari, known Hindi markers), and the grammar checker
        is the one that needs the morphology data. The retry suffix differs by
        layer — naming the specific words that failed is what turns a retry
        into a correction rather than a re-roll (DESIGN §5).
        """
        ok, reason = validator.validate(reply, self.max_reply_chars)
        if not ok:
            # Name the words when we know them, here too: a reply rejected for
            # Hindi comes back unchanged from a generic "answer in Sanskrit".
            hindi = validator.hindi_markers(reply)
            return False, reason, _correction_suffix(hindi)
        if self.grammar_checker is None:
            return True, "", ""
        try:
            findings = self.grammar_checker(reply)
        except Exception:
            # A checker fault must not cost the child an answer: the reply has
            # already passed the deterministic gate (FR-6.4).
            self.logger.exception("grammar check failed; accepting the reply")
            return True, "", ""
        if not findings:
            return True, "", ""
        words = self.grammar_checker.offending_words(findings)
        return (False, self.grammar_checker.reason(findings),
                _correction_suffix(words))

    def _try_cloud_fallback(self, message: str) -> str | None:
        if self._fallback_agent_factory is None:
            return None
        try:
            if self._fallback_agent is None:
                self._fallback_agent = self._fallback_agent_factory()
            reply = _limit_sentences(
                self._apply_lexicon(self._fallback_agent.converse(message)),
                self.max_sentences)
            ok, _, _ = self._check(reply)
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

    def _speak(self, text: str) -> str | None:
        """Deterministic speech path (DESIGN §1.4): synthesize, play without
        blocking (for barge-in), post playback_done when the speaker frees up.

        Returns the English gloss of ``text`` when glossing is on, else None.
        """
        self.logger.info("speak: %s", text)
        try:
            with self._tts_stage():
                wav, samplerate = self.tts.synthesize(text)
            self.robot.speaker_play(wav, samplerate, block=False)
        except Exception:
            self.logger.exception("TTS/playback failure (FR-6.4)")
            self.events.put(Event("playback_done"))
        else:
            threading.Thread(target=self._watch_playback, daemon=True).start()
        # Glossed only now, with the audio already playing: the translation is
        # a second model call, and it must not sit between a finished reply
        # and the speaker.
        return self._log_gloss(text)

    @contextmanager
    def _tts_stage(self):
        """Time synthesis into the turn log, when there is a turn to log."""
        if self.turn_logger is None or self.state != State.THINKING:
            yield
            return
        with self.turn_logger.stage("tts"):
            yield

    def _log_gloss(self, text: str) -> str | None:
        """Mirror the spoken line in English on the console (FR-7.2).

        A log convenience: it is never allowed to end a turn that already has
        its reply and its audio (FR-6.4).
        """
        if self.glosser is None:
            return None
        try:
            english = self.glosser.gloss(text)
        except Exception:
            self.logger.exception("English gloss failed; continuing")
            return None
        if english:
            self.logger.info("speak (en): %s", english)
        return english

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


def _correction_suffix(words: list[str]) -> str:
    """Retry instruction: name the offending words when there are any."""
    if not words:
        return prompts.CORRECTIVE_SUFFIX
    return prompts.WORD_CORRECTION_SUFFIX.format(words=", ".join(words))


def _limit_sentences(text: str, limit: int) -> str:
    """Keep at most ``limit`` sentences, terminator included.

    A reply with no terminator at all is left alone: truncating mid-clause
    would produce worse Sanskrit than the model wrote.
    """
    if limit <= 0:
        return text.strip()
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    if len(ends) <= limit:
        return text.strip()
    return text[:ends[limit - 1]].strip()


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
