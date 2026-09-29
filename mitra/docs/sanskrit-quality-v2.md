# Sanskrit output quality — work record

**Branch:** `sanskrit-quality-v2` | **Date:** 2026-08-22 | **Related:** [DESIGN.md](../DESIGN.md) §5 (v1.4), [REQUIREMENTS.md](../REQUIREMENTS.md) FR-3.5/3.6/3.8 (v1.6)

A condensed record of the session that added morphology-backed output checks: what
was wrong, what was measured, what was built, and what is still open. Numbers here
are reproducible with `scripts/eval_grammar.py` and `scripts/dry_run.py`.

---

## Summary — X → Y → Z

**X (the problem).** Mitra's replies were Sanskrit only in the sense that they were
written in Devanagari. Across three logged sessions of the same ten questions, between
10% and 45% of sentences were actually correct Sanskrit, and the errors repeated in
every run: Hindi words carrying Sanskrit endings — आज (*āja*) for अद्य "today", खेल
(*khela*) for क्रीडा "play", घर (*ghara*) for गृह "house", मक्खन/दूध for नवनीत/दुग्ध; verb
forms that exist in no language, such as करोष्यसि (a blend of करोषि and करिष्यसि) and
कुरुमि; and person disagreement, most often भवान् — a third-person honorific — paired
with a second-person verb (भवान् किं करोष्यसि instead of भवान् किं करोति). None of this
was visible to the system's only guardrail, which measured what *percentage* of the
reply was Devanagari script; Hindi shares that script, so every one of those sentences
scored 100% and was spoken to the child. A hand-written list of banned Hindi words was
added next, but it loses the race by construction: it needs a new entry for every new
word the model reaches for, and says nothing at all about invented forms.

**Y (what was implemented).** The check was inverted — from listing what Mitra may not
say to defining what it may. Every reply is now parsed against *vidyut*, an open
Sanskrit morphological lexicon of about 30 million inflected forms with Pāṇinian tags
(person, number, gender, case), and must pass three tests before it can be spoken:
every word must be an attested Sanskrit form; every word must be within Mitra's
declared vocabulary; and subject and verb must agree in person. The vocabulary is a
558-word everyday list — a published beginner's 500 words, plus the closed-class core
the list omits (कथम्, इति, the इदम् paradigm) and the words the system prompt's own
examples use — stored as **lemmas** rather than surface forms, so one entry admits a
whole paradigm: क्रीडति in the list allows क्रीडामि, क्रीडसि and क्रीडिष्यामि. When a reply
is rejected, the retry now names the offending word and asks for its replacement rather
than repeating a generic instruction. Alongside this, the Cologne dictionaries
(Monier-Williams, Apte) were indexed so that a human reviewing the robot's coined
object-names sees what the coined word actually means and what the dictionary offers
instead, and the debug console now prints an English translation under every Sanskrit
line so an operator who does not read Devanagari can see what was said.

**Z (the result).** On twenty hand-verified sentences in Mitra's own register the checks
reject none, and they catch all ten of the failure types drawn from the logs. In a live
session afterwards, eight of nine replies were correct Sanskrit — including अद्य अहं
पठिष्यामि ("today I shall read", correct future tense with the Sanskrit word for
"today", which all three earlier runs got wrong) and मह्यं गणितं रोचते ("mathematics is
pleasing to me", the idiomatic construction) — against one, one, and three of ten in
the previous runs. The single rejection fired on दूधम् (Hindi *dūdh*) and the corrected
reply used नवनीतम्. Not one Hindi word reached the speaker. The cost is about one second
of startup and no additional per-turn latency, since the translation runs after the
audio has already begun playing.

**Caveats.** This is one live sample, and a ten-turn run cannot resolve small
differences — a 50-question evaluation set is the next step. The disappearance of the
reciprocal-question template ("अहं X। भवान् किं Y?") is due to a separate change on the
same branch that truncates replies to one sentence, not to the morphology checks. And
the vocabulary check is strict by design: measured against an adult conversational
corpus it flags about a third of perfectly correct sentences, because that corpus's
register is far wider than a child's. It is calibrated for what Mitra says, not for
Sanskrit as a whole.

---

## The evidence that motivated it

Three sessions on commit `a50bfac`, same ten questions, no code change between them:

| | Run A | Run B | Run C |
|---|---|---|---|
| clean sentences | 3/20 | 2/20 | 9/20 |
| clean answer clauses | 1/10 | 1/10 | 3/10 |
| register used | भवान् | त्वम् | भवान् |
| invented verb forms | 3 | 6 | 1 |
| खेलानि for "play" | 3 turns | 3 turns | 2 turns |
| आज for "today" | ✓ | ✓ | ✓ |
| Hindi food loan | दाल | मक्खनम् | दालः |
| validator caught anything | 0 | 0 | 0 |

The spread (10% / 15% / 45%) is the finding that shaped the method: anything measured
on a ten-turn run is dominated by sampling noise at `temperature: 0.3`. What was stable
across all thirty turns — the Hindi lexical set, the invented forms, a guardrail that
never once fired — is what was worth fixing, and what is measurable deterministically.

## What was measured before anything was allowed to reject

`scripts/eval_grammar.py` scores each check two ways: against 924 human-authored
phrasebook sentences (every flag there is a **false positive**, and a false positive
costs a retry and can end in the safe fallback — i.e. Mitra goes quiet), and against a
gold set of 20 hand-verified sentences in Mitra's own register (the prompt's few-shot
examples and the orchestrator's fixed phrases).

| check | gold set (Mitra's register) | phrasebook (adult register) | logged replies |
|---|---|---|---|
| `unattested` | 0/20 false positives | ~17% | catches करोष्यसि, कुरुमि, मक्खनम्, दूधम् |
| `vocabulary` | 0/20 | ~32% | catches आज, घरे, खेलानि |
| `agreement` | 0/20 | 0.7% | catches भवान्…पठसि, त्वम्…अस्मि, अहं…चलन्ति |

Together: **0 false positives on the gold set, 10/10 known failures caught.**

### Calibration decisions, and why

* **Lemmas, not surface forms.** The source list gives verbs in the third person
  (क्रीडति) and Mitra speaks in the first (क्रीडामि); a surface list would have rejected
  its own vocabulary.
* **Built from stem readings, tested against all readings.** आज and अजा ("goat") share
  the krdanta root अज्. Matching on any shared lemma admits आज through अजा's entry;
  matching only on stems keeps it out while still allowing श्रोतुम् through श्रु.
* **Anusvāra normalisation.** The kosha indexes सङ्गीतम् and संगीतम् separately — the
  spelling the model happened to choose decided whether its own vocabulary matched.
* **Agreement runs on the segmenter's disambiguated parse**, not on every possible
  reading. भवतः is both a genitive and a verb form, so checking all readings flags
  अहं भवतः मित्रम् अस्मि — correct Sanskrit.
* **Split parts must be real words.** The segmenter will cut मक्खनम् into मक् + खनम्
  and कुरुमि into कुरुम् + इ, both halves "attested", so two invented words read as
  valid compounds. One-syllable parts are accepted only if the vocabulary vouches for
  them (मा in मास्तु, किम् in किमपि).
* **Rejected: a plain two-way compound split** as a fallback for long compounds the
  segmenter returns whole (जयनगरं). Measured — the kosha holds so many short forms that
  every invented word finds a "split": मक्खनम्, कुरुमि, करोष्यसि, दालः and even क्या
  passed. Long compounds now read as unattested; Mitra's register rarely produces them.
* **The dictionaries stay off the speaking path.** Substituting Apte's first answer for
  a coined name gives butter → नवनीतम् but also apple → तारा ("apple of the eye") and
  teacher → शास्. Apte carries idioms as well as words: a suggestion for a reviewer,
  not an authority over a child's answer.
* **"Replace", not "remove".** The first retry wording said *rewrite the sentence
  WITHOUT them*, and the model obeyed literally — asked about food it dropped the
  rejected दूधम् and answered मम प्रियं भोजनं अस्ति, "my favourite food exists". The
  wording now demands replacement; verified 3/3 and 4/4 runs afterwards.

## Live session after the change

| # | question | reply | verdict |
|---|---|---|---|
| 1 | What are you doing? | अहं कार्यं करोमि। | ✓ |
| 2 | Where do you live? | अहं नगरे अवस्थितः अस्मि। | ✓ correct, stilted |
| 3 | Do you play? | अहं क्रीडामि। | ✓ (was खेलानि करोमि in all earlier runs) |
| 4 | What's your favorite food? | दूधम् rejected → मम प्रियं भोजनं अस्ति। | ✗ contentless — retry wording, since fixed |
| 5 | What's your favorite subject? | मह्यं गणितं रोचते। | ✓ idiomatic |
| 6 | What are you reading? | अहं पुस्तकं पठामि। | ✓ |
| 7 | Do you play sports? | अहं क्रीडामि। | ✓ |
| 8 | What will you do today? | अद्य अहं पठिष्यामि। | ✓ future tense, अद्य not आज |
| 9 | Will you be my friend? | अहं भवतः मित्रम् अस्मि। | ✓ |

Latency, 9 turns: ASR p50 2.99 s (2.75–3.29), LLM p50 1.16 s, TTS p50 0.36 s, ≈4.5 s
per turn. Unchanged from the earlier runs — nothing here touches the speech path.

## What was built

```
src/sanskrit/analyzer.py      vidyut kosha: attestation, lemmas, morphology, segmentation
src/sanskrit/grammar.py       the three checks + the Checker the validator calls
src/lexicon/vocabulary.py     the lemma whitelist and how it is assembled
src/lexicon/vocabulary.jsonl  558 words (built artifact, committed)
src/lexicon/dictionary.py     Cologne MW/Apte lookups, for the review CLI
src/gloss.py                  debug-only English gloss of every spoken line (FR-7.2)
scripts/fetch_sanskrit_data.py   vidyut data (~78 MB) + Cologne sources (~55 MB)
scripts/build_vocabulary.py      the word list → lemmas (re-runnable from source)
scripts/build_dictionary.py      MW/Apte → SQLite index
scripts/eval_grammar.py          false-positive and catch rates on real corpora
scripts/dry_run.py               the whole pipeline with no robot and no audio
tests/test_sanskrit.py           gold set + logged failures, as a regression net
tests/test_dictionary.py, tests/test_gloss.py
```

Setup on a new machine:

```bash
pip install 'mitra[sanskrit]'
python3 scripts/fetch_sanskrit_data.py
python3 scripts/build_dictionary.py
python3 main.py --check          # reports the kosha, the lemma count, and the checks
```

Missing data disables the checks with one warning; nothing else changes (FR-6.4).

## Open items

1. **A 50-question evaluation set** — the single most valuable next thing. Ten turns
   cannot distinguish a real improvement from a good sample.
2. **दालः still leaks.** `sanskrit.ground_in_phrasebook: true` widens the vocabulary
   with the corpus, and the corpus itself uses the loanword (`किलो दालस्य कृते कति
   रूप्यकाणि ?`). दाल is also a genuine Sanskrit word in Monier-Williams, so no
   automatic check separates the senses. Options: leave it, set the flag false
   (stricter, more retries), or add दाल to `_HINDI_STEMS`.
3. **Sandhi before vowels** — भोजनं अस्ति should be भोजनम् अस्ति. No check sees it, and
   the Hindi VITS voice pronounces the two differently. A deterministic post-processor
   in the speak path would fix it outright.
4. **LLM warm-up at startup** is still unimplemented: the first turn of a cold session
   costs ~13.5 s.
5. **`TurnLogger.stage()` overwrites** on a retry, so a retried turn under-reports its
   LLM time (turn 4 above logged 0.695 s for ~2.0 s of work).
6. **Persona gender** — अवस्थितः is masculine while the prompt declares Mitra neuter
   (मित्रम्). Not a grammar error, but inconsistent across turns.
7. **Deferred resources** (see the resource plan): a gold answer bank for the ~150
   questions children actually ask, a Sanskrit reviewer for it and for the whitelist's
   content words, and a decision on cloud escalation instead of the safe fallback.
