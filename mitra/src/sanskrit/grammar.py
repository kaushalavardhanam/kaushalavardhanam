"""Grammar checks over a generated reply (DESIGN §5, FR-3.5).

Three checks, in the order a wrong sentence usually fails them:

``unattested``  the word is not a form of any Sanskrit word — करोष्यसि (a
                blend of करोषि and करिष्यसि), कुरुमि, मक्खनम्, दालः. These are
                the model inventing morphology, and nothing before this could
                see them: all four are 100% Devanagari.
``vocabulary``  the word is Sanskrit, but not a word this robot says — आज,
                घरे, खेलानि. Real forms of unrelated words, borrowed from
                Hindi by shape. See ``lexicon/vocabulary.py``.
``agreement``   subject and verb disagree in person — अहं ... चलन्ति ("I they
                move"), भवान् ... पठसि (भवान् is a third-person honorific and
                takes पठति), त्वम् ... अस्मि. Six of ten turns in one logged
                session carried this error.

Each check reports; none of them corrects. The retry policy is the
validator's, because it also owns what happens when a retry fails.

WHAT IS DELIBERATELY NOT CHECKED
--------------------------------
"Every sentence needs a finite verb" would catch अहं पुस्तकम् अध्ययनीयम्, but
Sanskrit drops अस्ति freely — मम नाम मित्रम् is correct and verbless — so the
rule fires on good sentences about as often as bad ones. Case government
(अहं X प्रियम् अस्मि) needs a parse this does not attempt: the ambiguity is
real, and a wrong parse would reject correct Sanskrit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Nominative subject pronouns and their person. Closed class, and the kosha is
# no help here: it analyses अहम् as a form of the root अह् before it reaches
# अस्मद्. Only nominatives are listed — मम/मह्यं/भवतः are oblique and are not
# the subject, so a sentence like मम नाम मित्रम् must not be read as one.
#
# भवान्/भवती are the trap this check exists for: they mean "you" but are
# grammatically THIRD person, so they take करोति, not करोषि.
_SUBJECT_PERSON = {
    "aham": "uttama", "AvAm": "uttama", "vayam": "uttama",
    "tvam": "madhyama", "yuvAm": "madhyama", "yUyam": "madhyama",
    "BavAn": "prathama", "BavatI": "prathama", "Bavantas": "prathama",
    "sas": "prathama", "sA": "prathama", "tat": "prathama",
    "ezas": "prathama", "ezA": "prathama", "etat": "prathama",
}

_PERSON_LABEL = {"uttama": "1st person", "madhyama": "2nd person",
                 "prathama": "3rd person"}

# Clause boundaries: danda, double danda, and the Latin terminators the model
# also produces. Commas split too — the model joins clauses with them, and a
# subject in one clause must not be matched against a verb in the next.
_CLAUSE = re.compile(r"[।॥?!,;]")


@dataclass
class Finding:
    check: str
    detail: str
    words: list[str] = field(default_factory=list)


def check(text: str, analyzer, vocabulary=None) -> list[Finding]:
    """Every problem found in ``text``; [] when the checks cannot run."""
    if analyzer is None or not analyzer.available:
        return []
    findings: list[Finding] = []

    # The vocabulary vouches for one-syllable parts of a split word; see
    # Analyzer.is_attested for why that guard has to exist.
    allow_part = vocabulary.short_ok if vocabulary is not None else None
    unattested = [w for w in analyzer.words(text)
                  if not analyzer.is_attested(w, allow_part)
                  and not (vocabulary and vocabulary.contains(w))]
    if unattested:
        findings.append(Finding(
            "unattested",
            "not Sanskrit words: " + ", ".join(dict.fromkeys(unattested)),
            list(dict.fromkeys(unattested))))

    if vocabulary is not None and vocabulary.available:
        outside = [w for w in vocabulary.unknown(text) if w not in unattested]
        if outside:
            findings.append(Finding(
                "vocabulary",
                "outside Mitra's vocabulary: " + ", ".join(outside),
                outside))

    findings.extend(agreement(text, analyzer))
    return findings


def agreement(text: str, analyzer) -> list[Finding]:
    """Subject-verb person disagreement, one clause at a time.

    Runs on the segmenter's disambiguated parse, not on every reading the
    kosha offers. Checking all readings flags अहं भवतः मित्रम् अस्मि — correct
    Sanskrit — because भवतः also happens to be a verb form. A clause the
    segmenter cannot parse is skipped rather than guessed at: this check
    rejects replies, and a false rejection costs the child an answer.
    """
    out: list[Finding] = []
    for clause in _CLAUSE.split(text or ""):
        if not analyzer.words(clause):
            continue
        parsed = analyzer.parse(clause)
        if not parsed:
            continue
        # Only nominatives are in the table, which is what keeps मम नाम
        # मित्रम् from reading as a first-person clause — the oblique forms
        # (मम, मह्यं, भवतः) are simply not subjects and are not listed. The
        # segmenter's own case tag is deliberately not consulted: it labels
        # भवान् accusative, and a check that rejects replies cannot lean on a
        # tag that is wrong on the commonest word in the corpus.
        subjects = {word: _SUBJECT_PERSON[analyzer.canonical(word)]
                    for word, _ in parsed
                    if analyzer.canonical(word) in _SUBJECT_PERSON}
        if len(set(subjects.values())) != 1:
            continue
        person = next(iter(subjects.values()))
        subject = next(iter(subjects))
        for word, analysis in parsed:
            if not analysis.is_verb or analysis.purusha == person:
                continue
            out.append(Finding(
                "agreement",
                f"{subject} is {_PERSON_LABEL[person]} but {word} is "
                f"{_PERSON_LABEL.get(analysis.purusha, analysis.purusha)}",
                [word]))
    return out


DEFAULT_CHECKS = ("unattested", "agreement", "vocabulary")


class Checker:
    """The checks the validator runs, bundled with what they need.

    ``checks`` selects which of them may reject a reply. All three are on by
    default; measured on 20 hand-verified sentences in Mitra's own register
    (the prompt's few-shot examples and the orchestrator's fixed phrases) they
    reject none of them, and they catch all ten of the logged failures. On the
    phrasebook — adult register, long compounds, and vocabulary far outside a
    beginner list — ``vocabulary`` fires on about a third of correct sentences
    and ``unattested`` on a sixth, so a deployment aiming at that register
    should turn those two down to reporting only.
    """

    def __init__(self, analyzer, vocabulary=None, checks=DEFAULT_CHECKS):
        self.analyzer = analyzer
        self.vocabulary = vocabulary
        self.checks = tuple(checks)

    @property
    def available(self) -> bool:
        return bool(self.analyzer is not None and self.analyzer.available
                    and self.checks)

    def __call__(self, text: str) -> list[Finding]:
        if not self.available:
            return []
        return [f for f in check(text, self.analyzer, self.vocabulary)
                if f.check in self.checks]

    @staticmethod
    def reason(findings: list[Finding]) -> str:
        return "; ".join(f.detail for f in findings)

    @staticmethod
    def offending_words(findings: list[Finding]) -> list[str]:
        words: list[str] = []
        for finding in findings:
            for word in finding.words:
                if word not in words:
                    words.append(word)
        return words
