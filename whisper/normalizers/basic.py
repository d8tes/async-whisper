import unicodedata
import re
import sys
try:
    import regex
    HAS_REGEX = True
except ImportError:
    HAS_REGEX = False

ADDITIONAL_DIACRITICS = {
    "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss", "ẞ": "SS",
    "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
}

_cat = unicodedata.category

_remove_brackets = re.compile(r"[<\[][^>\]]*[>\]]")
_remove_parens = re.compile(r"\(([^)]+?)\)")
_ws_multi = re.compile(r"\s+")
if HAS_REGEX:
    _split_letters = regex.compile(r"\X", regex.U)

REMOVE_MS_P = {c for c in range(sys.maxunicode) if _cat(chr(c))[0] in "MSP"}
DIACRITIC_DELETE = dict.fromkeys(i for i in range(sys.maxunicode) if _cat(chr(i)) == "Mn")
EXTRA_MAP = {ord(k): v for k, v in ADDITIONAL_DIACRITICS.items()}

def remove_symbols_and_diacritics(s: str, keep: str = "") -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    out_chars = []
    for c in s:
        if c in keep:
            out_chars.append(c)
        elif c in ADDITIONAL_DIACRITICS:
            out_chars.append(ADDITIONAL_DIACRITICS[c])
        elif _cat(c) == "Mn":
            continue
        elif _cat(c)[0] in "MSP":
            out_chars.append(" ")
        else:
            out_chars.append(c)
    return "".join(out_chars)

def remove_symbols(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return "".join(" " if _cat(c)[0] in "MSP" else c for c in s)

class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False, keep: str = ""):
        self.clean = (lambda t: remove_symbols_and_diacritics(t, keep)) if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str) -> str:
        if not s:
            return ""
        s = s.lower()
        s = _remove_brackets.sub("", s)
        s = _remove_parens.sub("", s)
        s = self.clean(s)
        if self.split_letters and HAS_REGEX:
            s = " ".join(_split_letters.findall(s))
        return _ws_multi.sub(" ", s).strip()
