"""Table-driven state machine tests (DESIGN §3, §9): FakeReachy + canned agent."""

import json
import time

from mitra.agent import prompts
from mitra.orchestrator import Event, Orchestrator, State

SA_REPLY = "मम नाम मित्रम्।"
EN_REPLY = "My name is Mitra, nice to meet you."


def test_wake_from_asleep_nods_and_greets(make_orchestrator, fake_robot, fake_tts):
    orch, _ = make_orchestrator()
    orch.handle_event(Event("wake"))
    assert orch.state == State.WAKING
    assert fake_robot.nods == 1
    assert fake_tts.spoken == [prompts.GREETING]
    orch.handle_event(Event("playback_done"))
    assert orch.state == State.LISTENING


def test_utterance_flows_to_spoken_reply(make_orchestrator, fake_tts):
    orch, agent = make_orchestrator(replies=[SA_REPLY])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "What is your name?"))
    assert orch.state == State.SPEAKING
    assert agent.calls == ["[lang=en] What is your name?"]
    assert fake_tts.spoken == [SA_REPLY]
    orch.handle_event(Event("playback_done"))
    assert orch.state == State.LISTENING


def test_invalid_reply_retries_with_corrective_suffix(make_orchestrator, fake_tts):
    orch, agent = make_orchestrator(replies=[EN_REPLY, SA_REPLY])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "hello"))
    assert len(agent.calls) == 2
    assert agent.calls[1].endswith(prompts.CORRECTIVE_SUFFIX)
    assert fake_tts.spoken == [SA_REPLY]


def test_double_failure_speaks_safe_fallback(make_orchestrator, fake_tts):
    orch, _ = make_orchestrator(replies=[EN_REPLY, EN_REPLY])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "hello"))
    assert fake_tts.spoken == [prompts.SAFE_FALLBACK]


def test_verified_lexicon_overrides_generated_name(make_orchestrator, fake_tts):
    # "apple" is seeded verified as सेवफलम्; the model generated a wrong name.
    vision_json = ('{"object_en": "apple", "name_sa_devanagari": "फलराजम्", '
                   '"name_iast": "phalarājam", "sentence_sa": "एतत् फलराजम् अस्ति।"}')
    orch, _ = make_orchestrator(replies=[vision_json])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "किम् एतत्?"))
    assert fake_tts.spoken == ["एतत् सेवफलम् अस्ति।"]


def test_new_object_recorded_unverified(make_orchestrator, fake_tts, lexicon):
    vision_json = ('{"object_en": "croissant", "name_sa_devanagari": "क्रुसाण्टम्", '
                   '"name_iast": "krusāṇṭam", "sentence_sa": "एतत् क्रुसाण्टम् अस्ति।"}')
    orch, _ = make_orchestrator(replies=[vision_json])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "what is this?"))
    assert fake_tts.spoken == ["एतत् क्रुसाण्टम् अस्ति।"]
    row = lexicon.lookup("croissant")
    assert row is not None and row["verified"] == 0


def test_end_session_speaks_farewell_then_sleeps(make_orchestrator, fake_tts):
    orch, agent = make_orchestrator(replies=["session_end"])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "goodbye"))
    assert orch.state == State.SPEAKING
    assert fake_tts.spoken == [prompts.FAREWELL]
    orch.handle_event(Event("playback_done"))
    assert orch.state == State.ASLEEP
    assert agent.resets == 1


def test_silence_timeout_returns_to_sleep(make_orchestrator):
    orch, agent = make_orchestrator(silence_timeout_s=30)
    orch.state = State.LISTENING
    orch._last_activity = time.monotonic() - 31
    orch.handle_event(Event("tick"))
    assert orch.state == State.ASLEEP
    assert agent.resets == 1


def test_barge_in_stops_playback(make_orchestrator, fake_robot):
    orch, _ = make_orchestrator()
    fake_robot.hold_playback = True
    orch.state = State.SPEAKING
    orch.handle_event(Event("wake"))
    assert fake_robot.stops == 1
    assert orch.state == State.LISTENING


def test_agent_exception_apologizes_and_keeps_session(make_orchestrator, fake_tts):
    class ExplodingAgent:
        def converse(self, message):
            raise RuntimeError("ollama down")

        def reset(self):
            pass

    orch, _ = make_orchestrator()
    orch.agent = ExplodingAgent()
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "hello"))
    assert fake_tts.spoken == [prompts.APOLOGY_RETRY]
    assert orch.state == State.SPEAKING  # → LISTENING on playback_done (FR-6.4)


def test_empty_transcript_asks_to_repeat(make_orchestrator, fake_tts):
    orch, agent = make_orchestrator()
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "   "))
    assert agent.calls == []
    assert fake_tts.spoken == [prompts.APOLOGY_RETRY]


def test_wake_ignored_while_listening(make_orchestrator, fake_robot):
    orch, _ = make_orchestrator()
    orch.state = State.LISTENING
    orch.handle_event(Event("wake"))
    assert orch.state == State.LISTENING
    assert fake_robot.nods == 0


def test_run_loop_stops_cleanly(make_orchestrator):
    import threading

    orch, _ = make_orchestrator()
    thread = threading.Thread(target=orch.run, daemon=True)
    thread.start()
    orch.stop()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_state_gestures_follow_transitions(make_orchestrator, fake_robot):
    orch, _ = make_orchestrator(replies=[SA_REPLY])
    orch.handle_event(Event("wake"))
    orch.handle_event(Event("playback_done"))          # WAKING -> LISTENING
    assert fake_robot.poses[-1] == "listening"
    orch.handle_event(Event("utterance", "hello"))     # -> THINKING -> SPEAKING
    assert "thinking" in fake_robot.poses
    assert fake_robot.poses[-1] == "neutral"           # face forward to speak
    orch.state = State.LISTENING
    orch._last_activity = time.monotonic() - 31
    orch.handle_event(Event("tick"))                   # timeout -> ASLEEP
    assert fake_robot.poses[-1] == "asleep"


def test_gestures_can_be_disabled(make_orchestrator, fake_robot):
    orch, _ = make_orchestrator(replies=[SA_REPLY], gestures=False)
    orch.handle_event(Event("wake"))
    orch.handle_event(Event("playback_done"))
    orch.handle_event(Event("utterance", "hello"))
    assert fake_robot.poses == []


def test_explain_in_english_allows_english_reply(make_orchestrator, fake_tts):
    english = "I said that the sun shines in the sky and gives us light."
    orch, agent = make_orchestrator(replies=[english])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "Can you explain that in English?"))
    assert fake_tts.spoken == [english]              # Devanagari check waived
    assert "[explain_in_english]" in agent.calls[0]  # model told it may switch


def test_plain_english_question_still_requires_sanskrit(make_orchestrator, fake_tts):
    orch, agent = make_orchestrator(replies=[EN_REPLY, EN_REPLY])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "Tell me about the sun."))
    assert "[explain_in_english]" not in agent.calls[0]
    assert fake_tts.spoken == [prompts.SAFE_FALLBACK]  # English reply rejected


def test_wake_plays_welcoming_emotion(make_orchestrator, fake_robot):
    orch, _ = make_orchestrator()
    orch.handle_event(Event("wake"))
    assert fake_robot.emotions == ["welcoming1"]


def test_sleep_plays_deep_sleep_emotion(make_orchestrator, fake_robot):
    orch, _ = make_orchestrator(silence_timeout_s=30)
    orch.state = State.LISTENING
    orch._last_activity = time.monotonic() - 31
    orch.handle_event(Event("tick"))
    assert "mini-deep-sleep" in fake_robot.emotions


def test_agent_exception_plays_confused_emotion(make_orchestrator, fake_robot):
    class ExplodingAgent:
        def converse(self, message):
            raise RuntimeError("ollama down")

        def reset(self):
            pass

    orch, _ = make_orchestrator()
    orch.agent = ExplodingAgent()
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "hello"))
    assert fake_robot.emotions == ["confused1"]


def test_emotions_disabled_alongside_gestures(make_orchestrator, fake_robot):
    orch, _ = make_orchestrator(replies=[SA_REPLY], gestures=False)
    orch.handle_event(Event("wake"))
    orch.handle_event(Event("playback_done"))
    orch.handle_event(Event("utterance", "hello"))
    assert fake_robot.emotions == []


def test_reply_truncated_to_one_sentence(make_orchestrator, fake_tts):
    """The stock trailing question carried most of the observed grammar
    errors (कथं भवतः?), so the answer is spoken without it."""
    orch, _ = make_orchestrator(replies=["अहं क्रीडामि। कथं भवतः?"])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "Do you play?"))
    assert fake_tts.spoken == ["अहं क्रीडामि।"]


def test_max_sentences_is_configurable(make_orchestrator, fake_tts):
    orch, _ = make_orchestrator(replies=["अहं क्रीडामि। अहं पठामि। अहं वदामि।"],
                                max_sentences=2)
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "Do you play?"))
    assert fake_tts.spoken == ["अहं क्रीडामि। अहं पठामि।"]


def test_single_sentence_reply_is_untouched(make_orchestrator, fake_tts):
    orch, _ = make_orchestrator(replies=["अहं पुस्तकं पठामि।"])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "What are you reading?"))
    assert fake_tts.spoken == ["अहं पुस्तकं पठामि।"]


def test_hindi_reply_is_retried_then_falls_back(make_orchestrator, fake_tts):
    """Pure-Devanagari Hindi passes the script check, so only the marker
    check can catch it; a failing retry must reach the safe fallback."""
    orch, agent = make_orchestrator(
        replies=["अहं आज पठामि।", "अहं आज पठामि।"])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "What will you do today?"))
    assert len(agent.calls) == 2                      # one corrective retry
    assert fake_tts.spoken == [prompts.SAFE_FALLBACK]


def test_hindi_retry_that_recovers_is_spoken(make_orchestrator, fake_tts):
    orch, _ = make_orchestrator(replies=["अहं आज पठामि।", "अद्य अहं पठामि।"])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "What will you do today?"))
    assert fake_tts.spoken == ["अद्य अहं पठामि।"]


class RecordingGlosser:
    """Stands in for mitra.gloss.Glosser: returns a fixed English line."""

    def __init__(self, english="My name is Mitra."):
        self.english = english
        self.seen: list[str] = []

    def gloss(self, text: str) -> str:
        self.seen.append(text)
        return self.english


def test_gloss_logs_english_and_lands_in_the_turn_log(make_orchestrator, tmp_path):
    from mitra.logging_subsystem import TurnLogger

    glosser = RecordingGlosser()
    orch, _ = make_orchestrator(replies=[SA_REPLY], glosser=glosser,
                                turn_logger=TurnLogger(tmp_path))
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "What is your name?"))
    assert glosser.seen == [SA_REPLY]
    record = json.loads((tmp_path / "turns.jsonl").read_text(encoding="utf-8"))
    assert record["reply"] == SA_REPLY
    assert record["reply_en"] == "My name is Mitra."
    assert "tts" in record["stages"]            # gloss did not replace the timing


def test_gloss_failure_never_breaks_a_turn(make_orchestrator, fake_tts):
    class BrokenGlosser:
        def gloss(self, text):
            raise RuntimeError("translator exploded")

    orch, _ = make_orchestrator(replies=[SA_REPLY], glosser=BrokenGlosser())
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "What is your name?"))
    assert fake_tts.spoken == [SA_REPLY]
    assert orch.state == State.SPEAKING


class WordChecker:
    """Stands in for mitra.sanskrit.grammar.Checker: rejects listed words."""

    def __init__(self, *bad):
        self.bad = set(bad)

    def __call__(self, text):
        from mitra.sanskrit.grammar import Finding

        hits = [w for w in self.bad if w in text]
        return [Finding("vocabulary", "outside Mitra's vocabulary: "
                        + ", ".join(hits), hits)] if hits else []

    @staticmethod
    def reason(findings):
        return "; ".join(f.detail for f in findings)

    @staticmethod
    def offending_words(findings):
        return [w for f in findings for w in f.words]


def test_grammar_rejection_retries_with_the_word_named(make_orchestrator, fake_tts):
    """घरे is 100% Devanagari and is not on any Hindi stoplist — only the
    morphology checks see it. The retry must name it, or the model re-rolls
    the same sentence."""
    orch, agent = make_orchestrator(
        replies=["अहं घरे अस्मि।", "अहं गृहे वसामि।"],
        grammar_checker=WordChecker("घरे"))
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "Where do you live?"))
    assert len(agent.calls) == 2
    assert "घरे" in agent.calls[1]                   # the retry names it
    assert fake_tts.spoken == ["अहं गृहे वसामि।"]


def test_hindi_marker_rejection_also_names_the_word(make_orchestrator):
    """The stoplist path gained the same treatment: खेलानि is caught by
    validator._HINDI_STEMS, and that retry now says which word to drop."""
    orch, agent = make_orchestrator(replies=["अहं खेलानि करोमि।", SA_REPLY])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "Do you play?"))
    assert "खेलानि" in agent.calls[1]


def test_grammar_failure_twice_falls_back(make_orchestrator, fake_tts):
    orch, _ = make_orchestrator(
        replies=["अहं घरे अस्मि।", "अहं घरे वसामि।"],
        grammar_checker=WordChecker("घरे"))
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "Where do you live?"))
    assert fake_tts.spoken == [prompts.SAFE_FALLBACK]


def test_a_broken_checker_never_costs_the_child_an_answer(make_orchestrator,
                                                          fake_tts):
    class BrokenChecker:
        def __call__(self, text):
            raise RuntimeError("kosha exploded")

    orch, _ = make_orchestrator(replies=[SA_REPLY], grammar_checker=BrokenChecker())
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "What is your name?"))
    assert fake_tts.spoken == [SA_REPLY]
