#!/usr/bin/env python3
"""Build the everyday-Sanskrit vocabulary whitelist (DESIGN §5).

Source: Open Pathshala's "500+ Sanskrit Words Every Beginner Must Know"
(https://openpathshala.com/common-sanskrit-words-with-translation), a
beginner word list in 28 everyday categories — pronouns, daily verbs, food,
school, house, animals, time. That is exactly the register Mitra speaks in.

WHY A WHITELIST AT ALL
----------------------
Attestation (``mitra.sanskrit.Analyzer``) rejects words that are not Sanskrit
forms — करोष्यसि, कुरुमि, मक्खनम्. It cannot reject Hindi words that happen to
be homographs of Sanskrit ones, and those are the ones the model actually
reaches for: आज (a real form of ājá), घरे (from a root ghṛ), खेलानि (a form of
खेल्). Every one of them scored a perfect Devanagari ratio and was spoken.
Listing the ~500 words Mitra is allowed to build sentences from closes that
hole in a way no blocklist can, because the blocklist grows with the model's
imagination and the whitelist does not.

The list is stored as LEMMAS, not surface forms: the source gives verbs in the
3rd person singular (क्रीडति) and Mitra speaks in the 1st (क्रीडामि), so the
whitelist would miss its own vocabulary if it stored what the page printed.
Lemmas come from the kosha, which is also how the runtime check resolves them
— the two sides agree by construction.

    python3 scripts/build_vocabulary.py                  # rebuild from the cached HTML
    python3 scripts/build_vocabulary.py --fetch          # re-download the page first
    python3 scripts/build_vocabulary.py --report         # show what did not resolve
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import mitra  # noqa: F401
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "mitra", ROOT / "src" / "__init__.py",
        submodule_search_locations=[str(ROOT / "src")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["mitra"] = module
    spec.loader.exec_module(module)

from mitra.sanskrit import Analyzer  # noqa: E402

# ---------------------------------------------------------------- supplement
#
# A beginner word list teaches content words; it does not list the closed-class
# machinery every sentence is built from. Measured against the phrasebook, the
# Open Pathshala list alone flags 87% of human-authored sentences — not because
# they are wrong, but because कथम्, इति, इदानीम्, भोः and the इदम् paradigm are
# not in it. Those are not judgment calls a Sanskrit reviewer needs to make, so
# they are added here rather than waiting on review.
#
# Two groups, and the distinction matters if this is ever trimmed:
#   pronouns/indeclinables — closed class, universal, safe to add
#   persona vocabulary     — the words prompts.py's own few-shot examples use.
#                            Without them the model is punished for following
#                            its instructions: मह्यं गणितं रोचते is an example
#                            IN THE PROMPT, and गणित is not in the word list.
#
# ``lemmas`` is given explicitly where deriving it from a surface form would
# drag in homographs: माम् resolves to asmad but also to मा ("do not") and म,
# and whitelisting those would admit words Mitra should never say.
CORE_SUPPLEMENT = [
    # (iast, english, tag, group, explicit lemmas or ())
    ("aham", "I (all forms: mām, mahyam, mama…)", "Pron", "core-pronoun", ("asmad",)),
    ("tvam", "you (all forms: tvām, tubhyam, tava…)", "Pron", "core-pronoun", ("yuzmad",)),
    ("idam", "this (ayam, iyam, asmin, asya…)", "Pron", "core-pronoun", ("idam",)),
    ("adaḥ", "that (yonder)", "Pron", "core-pronoun", ("adas",)),
    ("tat", "that (saḥ, sā, tasya, tasmin…)", "Pron", "core-pronoun", ("tad",)),
    ("etat", "this (eṣaḥ, eṣā, etasmin…)", "Pron", "core-pronoun", ("etad",)),
    ("yat", "which (relative)", "Pron", "core-pronoun", ("yad",)),
    ("bhavān", "you (honorific: bhavataḥ, bhavantam…)", "Pron", "core-pronoun", ("Bavat",)),
    ("katham", "how?", "Ind", "core-indeclinable", ()),
    ("kutaḥ", "from where?", "Ind", "core-indeclinable", ("kutas",)),
    ("kimartham", "why? / for what?", "Ind", "core-indeclinable", ("kimarTam",)),
    ("iti", "thus / end-quote", "Ind", "core-indeclinable", ()),
    ("evam", "thus / so", "Ind", "core-indeclinable", ()),
    ("tathā", "likewise", "Ind", "core-indeclinable", ()),
    ("idānīm", "now", "Ind", "core-indeclinable", ()),
    ("samyak", "properly / well", "Ind", "core-indeclinable", ()),
    ("kila", "indeed", "Ind", "core-indeclinable", ()),
    ("khalu", "surely", "Ind", "core-indeclinable", ()),
    ("nūnam", "certainly", "Ind", "core-indeclinable", ()),
    ("bhoḥ", "hey / vocative particle", "Ind", "core-indeclinable", ("Bos",)),
    ("mā", "do not (prohibitive)", "Ind", "core-indeclinable", ("mA",)),
    ("kati", "how many?", "Ind", "core-indeclinable", ()),
    ("sarva", "all / every", "Adj", "core-indeclinable", ()),
    ("kṛte", "for the sake of", "Ind", "core-indeclinable", ("kfte",)),
    ("namaḥ", "salutation (namaste)", "N", "persona", ("namas",)),
    ("nāma", "name", "N", "persona", ("nAman",)),
    ("priya", "dear / favourite", "Adj", "persona", ()),
    ("kuśalī", "well / in good health", "Adj", "persona", ("kuSalin",)),
    ("rocate", "is pleasing (mahyaṃ … rocate)", "ruc", "persona", ("ruc",)),
    ("bhāti", "shines", "bhā", "persona", ("BA",)),
    ("avagacchati", "understands", "ava+gam", "persona", ()),
    ("darśayati", "shows", "dṛś", "persona", ()),
    ("kṣamyatām", "forgive me", "kṣam", "persona", ("kzam",)),
    ("gaṇitam", "mathematics", "N", "persona", ()),
    ("saṃskṛtam", "Sanskrit (the language)", "N", "persona", ()),
    ("vijñānam", "science", "N", "persona", ()),
    ("navanītam", "butter", "N", "persona", ()),
    ("citram", "picture", "N", "persona", ()),
    ("vidyā", "learning / knowledge", "N", "persona", ()),
    ("krīḍā", "play / a game", "N", "persona", ()),
]

SOURCE_URL = "https://openpathshala.com/common-sanskrit-words-with-translation"
CACHED_HTML = ROOT / "data" / "openpathshala.html"
OUT_PATH = ROOT / "src" / "lexicon" / "vocabulary.jsonl"

_SECTION = re.compile(
    r'<h3[^>]*class="vocab-category-title"[^>]*>(.*?)</h3>(.*?)(?=<h3|\Z)', re.S)
_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_TAG = re.compile(r"<[^>]+>")
# "sundaraḥ (m) / sundarī (f)" — the parenthetical is a gloss, not the word.
_PAREN = re.compile(r"\([^)]*\)")


def text_of(cell: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", cell))).strip()


def parse(page: str) -> list[dict]:
    rows: list[dict] = []
    for heading, body in _SECTION.findall(page):
        section = text_of(heading)
        number, _, name = section.partition(".")
        header: list[str] = []
        for raw_row in _ROW.findall(body):
            cells = [text_of(c) for c in _CELL.findall(raw_row)]
            if not cells:
                continue
            if not header or cells[0].lower() == "devanagari":
                header = [c.lower() for c in cells]
                continue
            if len(cells) < 3:
                continue
            row = dict(zip(header, cells))
            devanagari = row.get("devanagari", cells[0])
            iast = row.get("iast", cells[1])
            # Third column is "Gender" for nouns, "Root" for verbs, and a
            # repeat of the gloss in a few sections. Kept verbatim as a tag;
            # nothing downstream depends on interpreting it.
            tag = cells[2] if len(cells) > 2 else ""
            english = row.get("english", cells[3] if len(cells) > 3 else "")
            for dev, ia in zip(*_align(devanagari, iast)):
                rows.append({
                    "iast": ia,
                    "devanagari_source": dev,
                    "tag": tag,
                    "english": english,
                    "section": f"{number.strip()}. {name.strip()}",
                })
    return rows


def _align(devanagari: str, iast: str) -> tuple[list[str], list[str]]:
    """Split "मात/ अम्बा" + "mātā / ambā" into aligned single words."""
    devs = [_PAREN.sub("", d).strip() for d in devanagari.split("/")]
    iasts = [_PAREN.sub("", i).strip() for i in iast.split("/")]
    if len(devs) != len(iasts):        # ragged row: keep the IAST, which is
        devs = [""] * len(iasts)       # what we transliterate from anyway
    return devs, iasts


_LINGA = {"m": "Pum", "f": "Stri", "n": "Napumsaka"}


def generate_forms(stem: str, tag: str) -> list[str]:
    """Decline a stem the kosha does not know, so its inflections are allowed.

    The kosha is a fixed stem list, not a dictionary: it is missing ordinary
    words (छात्रः, बालिका, छत्रम्) and every modern coinage (दूरवाणी,
    कारयानम्). Whitelisting only the nominative the page prints would then
    reject "बालिकायाः" while accepting "बालिका". vidyut.prakriya derives the
    full paradigm from the bare stem by Pāṇinian rule, which closes the gap
    for exactly the 40-odd words that need it.

    Returns [] for verbs and indeclinables — a verb needs a dhātu, not a
    nominal stem, and an indeclinable has only the one form already listed.
    """
    linga = _LINGA.get(tag.strip().lower()[:1]) if tag else None
    if linga is None:
        return []
    from vidyut.lipi import Scheme, transliterate
    from vidyut.prakriya import (Linga, Pada, Pratipadika, Vacana, Vibhakti,
                                 Vyakarana)

    # The page lists nouns in the nominative singular; strip that ending back
    # to the stem the derivation needs (छात्रः → छात्र, बालिका stays).
    if " " in stem.strip():          # "kadāpi na" is a phrase, not a stem
        return []
    slp = re.sub(r"[Hm]$", "", transliterate(stem, Scheme.Iast, Scheme.Slp1))
    vyakarana = Vyakarana()
    pratipadika = Pratipadika.basic(slp)
    vibhaktis = [Vibhakti.Prathama, Vibhakti.Dvitiya, Vibhakti.Trtiya,
                 Vibhakti.Caturthi, Vibhakti.Panchami, Vibhakti.Sasthi,
                 Vibhakti.Saptami, Vibhakti.Sambodhana]
    forms: set[str] = set()
    for vibhakti in vibhaktis:
        for vacana in (Vacana.Eka, Vacana.Dvi, Vacana.Bahu):
            try:
                derived = vyakarana.derive(Pada.Subanta(
                    pratipadika=pratipadika, linga=getattr(Linga, linga),
                    vibhakti=vibhakti, vacana=vacana))
            except Exception:
                continue
            for prakriya in derived:
                forms.add(transliterate(prakriya.text, Scheme.Slp1,
                                        Scheme.Devanagari))
    return sorted(forms)


def resolve(rows: list[dict], analyzer: Analyzer) -> list[dict]:
    """Attach Devanagari (from IAST) and kosha lemmas to every row.

    Devanagari is DERIVED from the IAST column rather than taken from the
    page: the source's own Devanagari carries typos (मातुलाानी for मातुलानी)
    and PDF exports of it break the conjuncts outright. IAST round-trips
    losslessly, so it is the authority here. Where the page's spelling
    disagrees with the derived one, the row is flagged rather than silently
    preferred either way.
    """
    from vidyut.lipi import Scheme, transliterate

    out = []
    for row in rows:
        iast = row["iast"]
        if not iast or not re.match(r"^[a-zāīūṛṝḷṃḥśṣñṅṇṭḍ' +-]+$", iast, re.I):
            continue
        # "vasanta-ṛtuḥ" is one compound written with a teaching hyphen.
        devanagari = transliterate(iast.replace("-", ""), Scheme.Iast,
                                   Scheme.Devanagari)
        lemmas = sorted(analyzer.lemmas(devanagari))
        entry = {
            "devanagari": devanagari,
            "iast": iast,
            "lemmas": lemmas,
            "english": row["english"],
            "tag": row["tag"],
            "section": row["section"],
            "source": "openpathshala",
        }
        if not lemmas:
            forms = generate_forms(iast.replace("-", ""), row["tag"])
            entry["forms"] = forms or [devanagari]
        if row["devanagari_source"] and row["devanagari_source"] != devanagari:
            entry["source_spelling"] = row["devanagari_source"]
        out.append(entry)
    return out


def supplement(analyzer: Analyzer) -> list[dict]:
    """CORE_SUPPLEMENT as vocabulary rows (see the table's own comment)."""
    from vidyut.lipi import Scheme, transliterate

    out = []
    for iast, english, tag, group, lemmas in CORE_SUPPLEMENT:
        devanagari = transliterate(iast, Scheme.Iast, Scheme.Devanagari)
        entry = {
            "devanagari": devanagari,
            "iast": iast,
            "lemmas": sorted(lemmas) if lemmas else sorted(analyzer.lemmas(devanagari)),
            "english": english,
            "tag": tag,
            "section": group,
            "source": "mitra",
        }
        if not entry["lemmas"]:
            entry["forms"] = [devanagari]
        out.append(entry)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="re-download the page")
    ap.add_argument("--report", action="store_true", help="list unresolved rows")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    if args.fetch or not CACHED_HTML.exists():
        print(f"fetching {SOURCE_URL}")
        req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            CACHED_HTML.parent.mkdir(parents=True, exist_ok=True)
            CACHED_HTML.write_bytes(resp.read())
    page = CACHED_HTML.read_text(encoding="utf-8")

    analyzer = Analyzer()
    if not analyzer.available:
        print("kosha unavailable — run scripts/fetch_sanskrit_data.py first",
              file=sys.stderr)
        return 1

    rows = resolve(parse(page), analyzer) + supplement(analyzer)
    seen, unique = set(), []
    for row in rows:
        if row["devanagari"] in seen:
            continue
        seen.add(row["devanagari"])
        unique.append(row)

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in unique:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    resolved = [r for r in unique if r["lemmas"]]
    generated = [r for r in unique if r.get("forms")]
    mismatched = [r for r in unique if "source_spelling" in r]
    print(f"{len(unique)} words -> {out_path.relative_to(ROOT)}")
    print(f"  {len(resolved)} resolved to lemmas ({len(resolved)/len(unique):.1%})")
    print(f"  {len(unique) - len(resolved)} unattested in the kosha, of which "
          f"{sum(1 for r in generated if len(r['forms']) > 1)} were declined "
          f"by prakriya ({sum(len(r['forms']) for r in generated)} forms)")
    print(f"  {len(mismatched)} differ from the page's own Devanagari spelling")
    if args.report:
        print("\nunattested (kept, but contribute no lemma to the whitelist):")
        for row in unique:
            if not row["lemmas"]:
                print(f"  {row['devanagari']:20} {row['iast']:20} {row['english'][:30]:30} {row['section']}")
        print("\nspelling mismatches (derived vs page):")
        for row in mismatched:
            print(f"  {row['devanagari']:20} vs {row['source_spelling']:20} ({row['iast']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
