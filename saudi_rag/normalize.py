"""Arabic text normalization and light stemming.

Arabic search breaks easily on spelling variation: hamza forms on alef,
taa marbuta vs haa, alef maqsura vs yaa, diacritics, tatweel and
Arabic-Indic digits. Normalizing both documents and queries the same way
fixes a large share of misses before any model is involved.
"""
import re

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_CHAR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})
_NON_WORD = re.compile(r"[^\w\s]")

# Function words that carry little retrieval signal (already normalized).
STOPWORDS = {
    "في", "من", "علي", "الي", "عن", "مع", "او", "ثم", "ما", "لا", "هل", "ان",
    "هذا", "هذه", "ذلك", "التي", "الذي", "كيف", "ماذا", "لماذا", "متي", "اين",
    "يا", "و", "انا", "عندي", "عندكم", "قد", "كل", "اي", "هو", "هي",
}

_PREFIXES = ("وبال", "وال", "بال", "كال", "فال", "لل", "ال")
_SUFFIXES = ("هما", "كم", "هم", "ها", "نا", "ات", "ون", "ين", "يه", "ه", "ي")


def normalize(text: str) -> str:
    """Return a normalized form of Arabic text (orthography only, no stemming)."""
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_CHAR_MAP)
    text = _NON_WORD.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def light_stem(token: str) -> str:
    """Strip one common prefix and one common suffix, keeping a stem of 3+ letters."""
    for p in _PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            token = token[len(p):]
            break
    else:
        # Leading conjunction waw on longer words ("وابي" -> "ابي").
        if token.startswith("و") and len(token) >= 5:
            token = token[1:]
    for s in _SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            token = token[: -len(s)]
            break
    return token


def tokenize(text: str, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    tokens = normalize(text).split()
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [light_stem(t) for t in tokens]
    return [t for t in tokens if t]
