"""Everyday-phrase retrieval for conversational grounding (DESIGN §4).

The lexicon (``store.py``) answers "what is this object called?". This answers
a different question: "how would a Sanskrit speaker phrase this kind of turn?"
Both feed the same agent call, and both exist for the same reason — an 8B
model with weak Sanskrit priors invents word-shaped things when left to
generate freely.

Corpus: संस्कृत व्यवहार साहस्री, ~1000 everyday sentences across 27 situations,
each with an English gloss and (via scripts/build_phrasebook.py) an IAST
transliteration.

Two retrieval keys, because transcripts arrive two ways. Whisper tags spoken
Sanskrit as English and romanises it, so "what is your name" has to match the
English gloss while "Sarvam kushalam" has to match the IAST. Matching on the
gloss alone returns noise for the second kind — which is the kind that matters
most here.

NOTE ON LICENCE: the book is published and copyrighted by Pallava Prakashan
and the transliteration is marked for personal study and research only. Keep
the derived JSONL out of any public repository; ship the loader, not the data.

Matching is IDF-weighted character-trigram cosine, and a matched row resolves
to what comes AFTER it in the book — a question to its answers, a line the user
just spoke to the reply that follows. Both replaced an earlier difflib
character-ratio match, which ranked by surface overlap and so let function
words decide: "What is your name?" retrieved "What is your scale of pay?"
while the corpus's own name row was discarded by a score ceiling. Measured on
tests/fixtures/retrieval_eval.jsonl (scripts/eval_retrieval.py), hit rate went
39% → 79% and precision 16% → 35%.

Still deliberately not embeddings: ~1000 rows and one query per turn. The
remaining failure is semantic, not lexical — "Do you listen to music?" matches
"Please listen to me." because they genuinely share words, and no character
model can tell them apart. A small CPU sentence-transformer is the next lever
if that matters; VRAM is not the obstacle it was assumed to be, since this
would run on CPU alongside ASR and TTS.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("mitra")

DEFAULT_PATH = "data/phrasebook.jsonl"

# IAST spells with diacritics (kuśalam, bhavān, kathaṃ); Whisper romanises
# without them (kushalam, bhavan, katham). Compared raw, those score too low
# to clear any useful floor, so both sides are folded to plain ASCII first.
# ś/ṣ → "sh" rather than "s" deliberately: that is how Whisper spells them.
_IAST_FOLD = str.maketrans({
    "ā": "a", "ī": "i", "ū": "u", "ṛ": "ri", "ṝ": "ri", "ḷ": "li",
    "ṃ": "m", "ṁ": "m", "ḥ": "h", "ñ": "n", "ṅ": "n", "ṇ": "n",
    "ṭ": "t", "ḍ": "d", "ś": "sh", "ṣ": "sh", "ē": "e", "ō": "o",
})


# Whisper writes IAST long vowels as doubled Latin ones — "naama" for nāma,
# "keem" for kim, "Bhavaan" for bhavān. The fold above turns ā into a single
# "a", so the two sides only meet if the doubles collapse too.
_DOUBLED_VOWELS = re.compile(r"(?:aa|ee|ii|oo|uu)")
_VOWEL_COLLAPSE = {"aa": "a", "ee": "i", "ii": "i", "oo": "u", "uu": "u"}


def _fold(text: str) -> str:
    """Lowercase, strip diacritics, collapse doubled vowels, drop punctuation.

    Also removes the danda, which build_phrasebook transliterates as a pipe
    ("dhanyavādaḥ |") and which never appears in a spoken transcript.
    """
    folded = (text or "").lower().translate(_IAST_FOLD)
    folded = re.sub(r"[^a-z0-9 ]+", " ", folded)
    folded = _DOUBLED_VOWELS.sub(lambda m: _VOWEL_COLLAPSE[m.group(0)], folded)
    return re.sub(r"\s+", " ", folded).strip()


# Stripped from the ENGLISH key only. These glosses are short ("What is your
# name?", "What is your scale of pay?") so function words make up most of the
# string, and a character-level match on them scores two unrelated questions as
# near-identical. Dropping them leaves the topical words to decide the match.
# The IAST key keeps every word: Sanskrit carries meaning in inflection, and
# there is no comparable stopword list for it here.
_EN_STOPWORDS = frozenset("""
a an the is are am was were be been being do does did done have has had
i me my mine you your yours he him his she her it its we us our they them
their this that these those to of in on at for from with by as and or but
not no so if then than there here what which who whom whose when where why
how will would shall should can could may might must
""".split())


def _content(text: str) -> str:
    """Folded English with function words removed (see _EN_STOPWORDS)."""
    kept = [w for w in text.split() if w not in _EN_STOPWORDS]
    return " ".join(kept) if kept else text


def _trigrams(text: str) -> set[str]:
    """Character trigrams over the folded string, spaces included.

    Character n-grams rather than words because neither side of the match can
    be trusted to word-break the same way: Whisper runs "nāma kim" together as
    "naamakim", and it misspells within words ("bhafatah" for bhavataḥ). Word
    tokens miss both; trigrams survive both, since one bad trigram costs one
    dimension instead of the whole token.
    """
    if not text:
        return set()
    padded = f" {text} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


class Phrasebook:
    """Nearest-phrase lookup over (english, iast, sanskrit, chapter) rows.

    Expected JSONL shape, one object per line:
        {"id": "01-0003", "chapter": "शिष्टाचारः",
         "chapter_en": "Common formulas", "sanskrit": "सुप्रभातम्।",
         "iast": "suprabhātam|", "english": "Good morning."}
    """

    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)
        self._rows: list[dict] = []
        if not self.path.exists():
            # Absent corpus must not break the pipeline: the orchestrator
            # simply runs ungrounded, exactly as it did before this existed.
            logger.warning("phrasebook not found at %s — running ungrounded",
                           self.path)
            return
        no_iast = 0
        with self.path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("phrasebook line %d is not valid JSON", line_no)
                    continue
                if not (row.get("english") and row.get("sanskrit")):
                    continue
                # Fold once at load, not once per row per query.
                row["_k_en"] = _fold(row.get("english", ""))
                row["_k_ia"] = _fold(row.get("iast", ""))
                if not row["_k_ia"]:
                    no_iast += 1
                row["_g_en"] = _trigrams(_content(row["_k_en"]))
                row["_g_ia"] = _trigrams(row["_k_ia"])
                # A row is a question if its gloss or its Sanskrit ends in "?".
                # This is what replaces the old score ceiling: the echo problem
                # is a question/answer role confusion, not a similarity value.
                row["_is_q"] = bool(
                    row["english"].strip().endswith("?")
                    or row["sanskrit"].strip().rstrip("।| ").endswith("?"))
                self._rows.append(row)
        self._build_index()
        logger.info("phrasebook: %d entries loaded", len(self._rows))
        if no_iast:
            logger.warning(
                "phrasebook: %d/%d rows have no IAST — romanised Sanskrit "
                "transcripts will not match these. Rebuild with "
                "indic-transliteration installed.", no_iast, len(self._rows))

    def sentences(self) -> list[str]:
        """The corpus's Sanskrit, for callers that want the text not the index
        (the vocabulary whitelist absorbs it — see lexicon/vocabulary.py)."""
        return [row["sanskrit"] for row in self._rows if row.get("sanskrit")]

    def _build_index(self) -> None:
        """Trigram IDF weights, row norms, and the two follow-on maps.

        IDF is what makes this discriminate where the old character-ratio
        match did not. Unweighted, "what is your name" and "what is your
        scale of pay" look nearly identical because the stopwords dominate
        the string; weighted, the shared "what is your" costs almost nothing
        and the rare trigrams of "name" carry the match.
        """
        df: dict[str, int] = defaultdict(int)
        for row in self._rows:
            for gram in row["_g_en"] | row["_g_ia"]:
                df[gram] += 1
        total = len(self._rows) or 1
        self._idf = {g: math.log(total / (1 + n)) + 1.0 for g, n in df.items()}

        for row in self._rows:
            for field in ("en", "ia"):
                grams = row[f"_g_{field}"]
                row[f"_n_{field}"] = math.sqrt(
                    sum(self._idf.get(g, 0.0) ** 2 for g in grams)) or 1.0

        # Answers follow their question until the next question in the same
        # chapter (verified against the corpus: 02-0016 "What is your name?"
        # is answered by 02-0018, 02-0022 by 02-0024/25, 02-0044 by 02-0045).
        self._answers: dict[int, list[int]] = {}
        for i, row in enumerate(self._rows):
            if not row["_is_q"]:
                continue
            block: list[int] = []
            j = i + 1
            # The corpus groups question variants together before answering
            # them once — 02-0022 "What do you do? (masc.)" and 02-0023 "(fem.)"
            # share the answers at 02-0024/25. Step over the variants first, or
            # the masculine form ends up with no answer block at all.
            while (j < len(self._rows)
                   and self._rows[j].get("chapter") == row.get("chapter")
                   and self._rows[j]["_is_q"]):
                j += 1
            while j < len(self._rows):
                nxt = self._rows[j]
                if nxt.get("chapter") != row.get("chapter") or nxt["_is_q"]:
                    break
                # Skip bare vocabulary continuations ("Officer;", "Typist")
                # — they are word lists, not sentences to imitate.
                if " " in nxt["_k_en"] or " " in nxt["_k_ia"]:
                    block.append(j)
                if len(block) >= 3:
                    break
                j += 1
            if block:
                self._answers[i] = block

        # Statement rows need the same treatment for a different reason. The
        # corpus holds both halves of a conversation, so a near-perfect match
        # on a statement means the user just said that very line — "Sarvam
        # kushalam" matches सर्वं कुशलम्। exactly. Handing it back is the echo
        # the old score ceiling discarded matches to avoid. Advancing to what
        # follows it in the book (कः विशेषः ? — "what news?") keeps the row's
        # value and drops the echo.
        self._continuations: dict[int, list[int]] = {}
        for i, row in enumerate(self._rows):
            if row["_is_q"]:
                continue
            block = []
            for j in range(i + 1, len(self._rows)):
                nxt = self._rows[j]
                if nxt.get("chapter") != row.get("chapter"):
                    break
                if " " in nxt["_k_en"] or " " in nxt["_k_ia"]:
                    block.append(j)
                if len(block) >= 2:
                    break
            if block:
                self._continuations[i] = block

    def _score(self, query_grams: set[str], row: dict, field: str) -> float:
        """IDF-weighted cosine between a query and one of a row's two keys."""
        grams = row[f"_g_{field}"]
        if not grams or not query_grams:
            return 0.0
        idf = self._idf
        shared = query_grams & grams
        if not shared:
            return 0.0
        dot = sum(idf.get(g, 0.0) ** 2 for g in shared)
        q_norm = math.sqrt(sum(idf.get(g, 0.0) ** 2 for g in query_grams)) or 1.0
        return dot / (q_norm * row[f"_n_{field}"])

    def _resolve(self, score: float, i: int, echo: float = 0.72) -> list[int]:
        """Which rows a match at index ``i`` should actually contribute.

        A question yields its answers; a statement the user has evidently just
        spoken yields what follows it; anything else yields itself.
        """
        if self._rows[i]["_is_q"]:
            return self._answers.get(i, [i])
        if score >= echo:
            return self._continuations.get(i, [i])
        return [i]

    def count(self) -> int:
        return len(self._rows)

    def similar(self, query: str, k: int = 3, floor: float = 0.30,
                echo: float = 0.72) -> list[dict]:
        """The k most relevant rows, best first, matched on English gloss OR IAST.

        A row scores as the better of its two keys, so one index serves both
        "how are you" and "kathamasi" without needing to know which kind of
        transcript arrived.

        Utterances often carry more than one sentence ("Namaste. Kathangasi.")
        while rows hold exactly one, and a long query dilutes the score against
        every short row. So the query is split on sentence punctuation and each
        part scored separately, with the row keeping its best part.

        ``floor`` matters more than ``k``: three bad matches are worse than
        none, because the model imitates whatever register it is handed. Below
        the floor nothing is attached and the turn runs ungrounded, which is
        the correct outcome for the many turns this corpus cannot answer.

        Questions are resolved to their ANSWERS. This corpus holds both halves
        of a conversation, so the closest row to "bhavatah nama kim" is that
        very question, भवतः नाम किं ? — handing it back as reference taught the
        model to ask the question again instead of answering it. Returning the
        answer block instead (मम नाम fill ।) gives the model the one thing it
        is worst at inventing: a correctly inflected Sanskrit reply.
        """
        parts = [p for p in (_fold(seg) for seg in re.split(r"[.?!।]", query or ""))
                 if len(p) >= 3]
        whole = _fold(query)
        if whole and whole not in parts:
            parts.append(whole)
        if not parts or not self._rows:
            return []

        # One query representation per key: content words only against the
        # English gloss, the full string against the IAST (see _EN_STOPWORDS).
        grams_by_field = {
            "en": [_trigrams(_content(part)) for part in parts],
            "ia": [_trigrams(part) for part in parts],
        }
        scored: list[tuple[float, int]] = []
        for i, row in enumerate(self._rows):
            best = 0.0
            for field in ("en", "ia"):
                for grams in grams_by_field[field]:
                    score = self._score(grams, row, field)
                    if score > best:
                        best = score
            if best >= floor:
                scored.append((best, i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))

        # Round-robin across the top matches rather than draining one match's
        # answer block first. A single wrong top hit would otherwise fill every
        # slot with rows from one unrelated exchange; spreading the slots means
        # a near-miss at rank 1 still leaves room for the right row at rank 2.
        blocks = [self._resolve(score, i, echo) for score, i in scored[:k]]
        out: list[dict] = []
        seen: set[int] = set()
        for depth in range(3):
            for block in blocks:
                if depth >= len(block):
                    continue
                j = block[depth]
                if j in seen:
                    continue
                seen.add(j)
                out.append(self._rows[j])
                if len(out) >= k:
                    return out
        return out

    def by_chapter(self, chapter: str) -> list[dict]:
        """All rows from one situation, e.g. for seeding a scripted demo."""
        return [r for r in self._rows if r.get("chapter") == chapter]
