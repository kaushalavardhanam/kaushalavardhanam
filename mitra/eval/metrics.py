"""Automated per-reply component metrics for the MITRA coherence check.

Metrics (computed over a group of replies, e.g. one category or one depth):

    grammar_score_mean        mean of the verifier's grammar score
    recognition_rate          share of Devanagari words found by a word lookup
                              (Vidyut), pooled over all replies
    pronoun_verb_errors       total pronoun-verb agreement errors (heuristic)
    replies_with_hindi        replies flagged by the Hindi detector
    replies_with_latin        replies containing Latin-script letters
    replies_with_4gram_repeat replies in which some word 4-gram occurs twice+
    median_words              median words per non-empty reply
    empty_replies             replies that are None, empty or whitespace only

The verifiers are injected as callables so the metrics stay testable:

    grammar_fn(text) -> float
    hindi_fn(text)   -> bool
    word_lookup(word) -> bool     (see vidyut_word_lookup)

Pronoun-verb agreement is a heuristic: for each unambiguous pronoun in a
sentence (aham, vayam, tvam, yuyam, sah, sa, ...) the person/number implied by
verb-like endings in the same sentence is compared. A sentence with a pronoun
and at least one verb-like word, none of which agrees, counts as one error.
Noun forms ending like verbs (e.g. ramah) can cause false positives/negatives;
treat the count as a screening signal, not ground truth.

Usage (from the mitra/ directory):

    python -m eval.metrics                # per-category table from harness records
    python -m eval.metrics --path X.jsonl

The Vidyut lookup reads the data directory from MITRA_VIDYUT_DATA. The Vidyut
API call used here (vidyut.kosha.Kosha(...).contains) was written without
being run against an installed Vidyut; check it against your version.
"""

import argparse
import os
import re
import statistics
import sys
from collections import defaultdict

# Devanagari block without the danda (U+0964) and double danda (U+0965)
# and without Devanagari digits (U+0966-U+096F).
_DEVANAGARI_WORD = "[\u0900-\u0963\u0970-\u097F]+"
_LATIN_WORD = "[A-Za-z\u00C0-\u024F]+"
_TOKEN_RE = re.compile(_DEVANAGARI_WORD + "|" + _LATIN_WORD + "|[0-9\u0966-\u096F]+")
_DEVANAGARI_RE = re.compile(_DEVANAGARI_WORD)
_LATIN_RE = re.compile("[A-Za-z\u00C0-\u024F]")
_SENTENCE_SPLIT_RE = re.compile("[।॥.!?\n]+")

# Pronoun -> person/number code.
PRONOUNS = {
    "अहम्": "1sg", "अहं": "1sg",
    "आवाम्": "1du", "आवां": "1du",
    "वयम्": "1pl", "वयं": "1pl",
    "त्वम्": "2sg", "त्वं": "2sg",
    "युवाम्": "2du", "युवां": "2du",
    "यूयम्": "2pl", "यूयं": "2pl",
    "सः": "3sg", "सा": "3sg", "एषः": "3sg", "एषा": "3sg",
    "तौ": "3du", "ताः": "3pl",
}

# Verb endings, longest/most specific first.
VERB_ENDINGS = [
    ("न्ति", "3pl"), ("न्ते", "3pl"),
    ("मि", "1sg"), ("वः", "1du"), ("मः", "1pl"),
    ("सि", "2sg"), ("थः", "2du"), ("थ", "2pl"),
    ("ति", "3sg"), ("ते", "3sg"), ("तः", "3du"),
]


def tokenize(text):
    """Return word tokens (Devanagari, Latin, digits); punctuation is dropped."""
    return _TOKEN_RE.findall(text or "")


def devanagari_words(text):
    return _DEVANAGARI_RE.findall(text or "")


def is_empty(text):
    return text is None or not str(text).strip()


def word_count(text):
    return len(tokenize(text))


def has_latin(text):
    """True if the text contains any Latin-script letter."""
    return bool(_LATIN_RE.search(text or ""))


def repeated_4grams(text):
    """Number of extra occurrences of word 4-grams within one reply.

    "a b c d a b c d" has one 4-gram (a b c d) seen twice -> 1.
    """
    tokens = tokenize(text)
    seen = {}
    for i in range(len(tokens) - 3):
        gram = tuple(tokens[i:i + 4])
        seen[gram] = seen.get(gram, 0) + 1
    return sum(c - 1 for c in seen.values() if c > 1)


def verb_person_number(word):
    """Person/number implied by a verb-like ending, or None."""
    if word in PRONOUNS:
        return None
    for ending, code in VERB_ENDINGS:
        if word.endswith(ending) and len(word) > len(ending):
            return code
    return None


def pronoun_verb_errors(text):
    """Count pronouns that disagree with every verb-like word in their sentence."""
    errors = 0
    for sentence in _SENTENCE_SPLIT_RE.split(text or ""):
        words = devanagari_words(sentence)
        pronouns = [PRONOUNS[w] for w in words if w in PRONOUNS]
        if not pronouns:
            continue
        verbs = [c for c in (verb_person_number(w) for w in words) if c]
        if not verbs:
            continue
        for code in pronouns:
            if code not in verbs:
                errors += 1
    return errors


def recognition_counts(text, word_lookup):
    """Return (recognized, total) Devanagari words according to word_lookup."""
    words = devanagari_words(text)
    return sum(1 for w in words if word_lookup(w)), len(words)


def vidyut_word_lookup(data_dir=None):
    """Return a word_lookup(word) -> bool backed by a Vidyut Kosha.

    data_dir defaults to the MITRA_VIDYUT_DATA environment variable; the
    kosha is expected under <data_dir>/kosha. Raises RuntimeError if Vidyut
    or the data is unavailable.
    """
    data_dir = data_dir or os.environ.get("MITRA_VIDYUT_DATA")
    if not data_dir:
        raise RuntimeError("Set MITRA_VIDYUT_DATA to the Vidyut data directory")
    try:
        from vidyut.kosha import Kosha
    except ImportError as exc:
        raise RuntimeError("vidyut is not installed") from exc
    kosha = Kosha(os.path.join(data_dir, "kosha"))
    return lambda word: bool(kosha.contains(word))


def compute_metrics(replies, grammar_scores=None, hindi_flags=None, word_lookup=None):
    """Aggregate metrics for a list of reply texts.

    grammar_scores / hindi_flags are optional lists aligned with replies
    (None entries are ignored). word_lookup is optional; without it,
    recognition_rate is None.
    """
    replies = list(replies)
    grammar_scores = grammar_scores if grammar_scores is not None else [None] * len(replies)
    hindi_flags = hindi_flags if hindi_flags is not None else [None] * len(replies)

    non_empty = [r for r in replies if not is_empty(r)]
    scores = [
        s for r, s in zip(replies, grammar_scores) if s is not None and not is_empty(r)
    ]
    recognized = total = 0
    if word_lookup is not None:
        for r in non_empty:
            rec, tot = recognition_counts(r, word_lookup)
            recognized += rec
            total += tot

    return {
        "n_replies": len(replies),
        "grammar_score_mean": sum(scores) / len(scores) if scores else None,
        "recognition_rate": (recognized / total) if (word_lookup is not None and total) else None,
        "pronoun_verb_errors": sum(pronoun_verb_errors(r) for r in non_empty),
        "replies_with_hindi": sum(1 for f in hindi_flags if f),
        "replies_with_latin": sum(1 for r in non_empty if has_latin(r)),
        "replies_with_4gram_repeat": sum(1 for r in non_empty if repeated_4grams(r) > 0),
        "median_words": statistics.median(word_count(r) for r in non_empty) if non_empty else None,
        "empty_replies": len(replies) - len(non_empty),
    }


def metrics_from_records(records, word_lookup=None):
    """Aggregate metrics for harness records, using their stored verifier fields."""
    return compute_metrics(
        [r.get("text") for r in records],
        grammar_scores=[r.get("grammar_score") for r in records],
        hindi_flags=[r.get("hindi_flag") for r in records],
        word_lookup=word_lookup,
    )


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main(argv=None) -> int:
    from eval.harness import _log_path, load_records

    parser = argparse.ArgumentParser(description="Per-category component metrics")
    parser.add_argument("--path", help="harness JSONL path")
    parser.add_argument("--vidyut", action="store_true", help="compute recognition rate with Vidyut")
    args = parser.parse_args(argv)

    records = load_records(args.path)
    if not records:
        print("No records at", _log_path(args.path))
        return 0
    lookup = vidyut_word_lookup() if args.vidyut else None

    groups = defaultdict(list)
    for r in records:
        groups[str(r.get("category"))].append(r)
    keys = ["n_replies", "grammar_score_mean", "recognition_rate", "pronoun_verb_errors",
            "replies_with_hindi", "replies_with_latin", "replies_with_4gram_repeat",
            "median_words", "empty_replies"]
    print("category".ljust(20) + " " + " ".join(k[:14].rjust(14) for k in keys))
    for category in sorted(groups):
        m = metrics_from_records(groups[category], word_lookup=lookup)
        print(category[:20].ljust(20) + " " + " ".join(_fmt(m[k]).rjust(14) for k in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())