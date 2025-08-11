import os
import re
import json
from fractions import Fraction
from typing import List, Optional, Iterator, Union, Match
from more_itertools import windowed
from .basic import remove_symbols_and_diacritics

class EnglishNumberNormalizer:
    def __init__(self):
        self.zeros = {"o","oh","zero"}
        self.ones = {
            n: i for i, n in enumerate([
                "one","two","three","four","five","six","seven","eight","nine",
                "ten","eleven","twelve","thirteen","fourteen","fifteen",
                "sixteen","seventeen","eighteen","nineteen"], start=1)
        }
        self.ones_plural = {"sixes" if n=="six" else n+"s": (v,"s") for n,v in self.ones.items()}
        self.ones_ordinal = {
            "zeroth": (0,"th"), "first": (1,"st"), "second": (2,"nd"), "third": (3,"rd"),
            "fifth": (5,"th"), "twelfth": (12,"th"),
            **{n+("h" if n.endswith("t") else "th"):(v,"th") for n,v in self.ones.items() if v>3 and v!=5 and v!=12}
        }
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}
        self.tens = {
            "twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,
            "eighty":80,"ninety":90
        }
        self.tens_plural = {n.replace("y","ies"):(v,"s") for n,v in self.tens.items()}
        self.tens_ordinal = {n.replace("y","ieth"):(v,"th") for n,v in self.tens.items()}
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}
        self.multipliers = {
            "hundred":100,"thousand":1_000,"million":1_000_000,"billion":1_000_000_000,
            "trillion":1_000_000_000_000,"quadrillion":1_000_000_000_000_000,
            "quintillion":1_000_000_000_000_000_000
        }
        self.multipliers_plural = {n+"s": (v,"s") for n,v in self.multipliers.items()}
        self.multipliers_ordinal = {n+"th": (v,"th") for n,v in self.multipliers.items()}
        self.multipliers_suffixed = {**self.multipliers_plural, **self.multipliers_ordinal}
        self.decimals = {*self.ones, *self.tens, *self.zeros}
        self.preceding_prefixers = {"minus":"-","negative":"-","plus":"+", "positive":"+"}
        self.following_prefixers = {"pound":"£","pounds":"£","euro":"€","euros":"€",
                                   "dollar":"$","dollars":"$","cent":"¢","cents":"¢"}
        self.prefixes = set(self.preceding_prefixers.values()) | set(self.following_prefixers.values())
        self.suffixers = {"per":{"cent":"%"}, "percent":"%"}
        self.specials = {"and","double","triple","point"}
        self.words = set().union(
            self.zeros, self.ones, self.ones_suffixed, self.tens, self.tens_suffixed,
            self.multipliers, self.multipliers_suffixed, self.preceding_prefixers,
            self.following_prefixers, self.suffixers, self.specials
        )
        self.literal_words = {"one","ones"}
        self._dec_re = re.compile(r"^\d+(\.\d+)?$")

    def _to_fraction(self, s: str) -> Optional[Fraction]:
        try:
            return Fraction(s)
        except ValueError:
            return None

    def process_words(self, words: List[str]) -> Iterator[str]:
        prefix, value, skip = None, None, False
        if not words:
            return
        for prev, current, nxt in windowed([None]+words+[None], 3):
            if skip:
                skip = False
                continue
            next_is_numeric = nxt is not None and self._dec_re.match(nxt)
            has_prefix = bool(current and current[0] in self.prefixes)
            current_wo_prefix = current[1:] if has_prefix else current
            if self._dec_re.match(current_wo_prefix):
                f = self._to_fraction(current_wo_prefix)
                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        value += current
                        continue
                    else:
                        yield self._out(value, prefix)
                        prefix = None
                prefix = current[0] if has_prefix else prefix
                value = f.numerator if f and f.denominator == 1 else current_wo_prefix
            elif current not in self.words:
                if value is not None:
                    yield self._out(value, prefix)
                    prefix = None
                yield current
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                value = self._handle_digit(value, self.ones[current], prev)
            elif current in self.ones_suffixed:
                yield from self._handle_suffixed(value, prev, *self.ones_suffixed[current])
                value = None
            elif current in self.tens:
                value = self._handle_tens(value, self.tens[current])
            elif current in self.tens_suffixed:
                yield from self._handle_tens_suffixed(value, *self.tens_suffixed[current])
            elif current in self.multipliers:
                value = self._handle_multiplier(value, self.multipliers[current], prefix)
            elif current in self.multipliers_suffixed:
                yield from self._handle_multiplier_suffixed(value, *self.multipliers_suffixed[current])
                value = None
            elif current in self.preceding_prefixers:
                if value is not None:
                    yield self._out(value, prefix)
                    prefix = None
                if nxt in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield current
            elif current in self.following_prefixers:
                if value is not None:
                    yield self._out(value, self.following_prefixers[current])
                    value = None
                else:
                    yield current
            elif current in self.suffixers:
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if nxt in suffix:
                            yield self._out(str(value) + suffix[nxt], prefix)
                            skip = True
                        else:
                            yield self._out(value, prefix)
                            yield current
                    else:
                        yield self._out(str(value) + suffix, prefix)
                    value = None
                else:
                    yield current
            elif current in self.specials:
                if nxt not in self.words and not next_is_numeric:
                    if value is not None:
                        yield self._out(value, prefix)
                        value = None
                    yield current
                elif current == "and":
                    if prev not in self.multipliers:
                        if value is not None:
                            yield self._out(value, prefix)
                            value = None
                        yield current
                elif current in {"double", "triple"}:
                    if nxt in self.ones or nxt in self.zeros:
                        repeats = 2 if current == "double" else 3
                        ones = self.ones.get(nxt, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield self._out(value, prefix)
                            value = None
                        yield current
                elif current == "point":
                    if nxt in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
            if value is not None:
                yield self._out(value, prefix)

    def _out(self, val, pre):
        return (pre or "") + str(val)

    def _handle_digit(self, value, d, prev):
        if value is None:
            return d
        if isinstance(value, str) or prev in self.ones:
            if prev in self.tens and d < 10:
                assert value[-1] == "0"
                return value[:-1] + str(d)
            return str(value) + str(d)
        if d < 10:
            if value % 10 == 0:
                return value + d
            return str(value) + str(d)
        if value % 100 == 0:
            return value + d
        return str(value) + str(d)

    def _handle_suffixed(self, value, prev, num, suf):
        if value is None:
            yield str(num) + suf
        elif isinstance(value, str) or prev in self.ones:
            if prev in self.tens and num < 10:
                assert value[-1] == "0"
                yield value[:-1] + str(num) + suf
            else:
                yield str(value) + str(num) + suf
        elif num < 10:
            if value % 10 == 0:
                yield str(value + num) + suf
            else:
                yield str(value) + str(num) + suf
        else:
            if value % 100 == 0:
                yield str(value + num) + suf
            else:
                yield str(value) + str(num) + suf

    def _handle_tens(self, value, t):
        if value is None:
            return t
        if isinstance(value, str):
            return str(value) + str(t)
        return value + t if value % 100 == 0 else str(value) + str(t)

    def _handle_tens_suffixed(self, value, t, suf):
        if value is None:
            yield str(t) + suf
        elif isinstance(value, str):
            yield str(value) + str(t) + suf
        else:
            if value % 100 == 0:
                yield str(value + t) + suf
            else:
                yield str(value) + str(t) + suf

    def _handle_multiplier(self, value, mult, prefix):
        if value is None:
            return mult
        if isinstance(value, str) or value == 0:
            f = self._to_fraction(value)
            p = f * mult if f else None
            if f and p.denominator == 1:
                return p.numerator
            yield self._out(value, prefix)
            return mult
        before = value // 1000 * 1000
        return before + (value % 1000) * mult

    def _handle_multiplier_suffixed(self, value, mult, suf):
        if value is None:
            yield str(mult) + suf
        elif isinstance(value, str):
            f = self._to_fraction(value)
            p = f * mult if f else None
            if f and p.denominator == 1:
                yield str(p.numerator) + suf
            else:
                yield value
                yield str(mult) + suf
        else:
            before = value // 1000 * 1000
            val = before + (value % 1000) * mult
            yield str(val) + suf

    def preprocess(self, s: str) -> str:
        results = []
        segments = re.split(r"\band\s+a\s+half\b", s or "")
        for i, seg in enumerate(segments):
            if not seg.strip():
                continue
            results.append(seg)
            if i != len(segments) - 1:
                lw = seg.rsplit(maxsplit=2)[-1]
                results.append("point five" if lw in self.decimals or lw in self.multipliers else "and a half")
        s = " ".join(results)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)
        return s

    def postprocess(self, s: str) -> str:
        def combine_cents(m: Match) -> str:
            try:
                return f"{m.group(1)}{m.group(2)}.{int(m.group(3)):02d}"
            except Exception:
                return m.string

        def extract_cents(m: Match) -> str:
            try:
                return f"¢{int(m.group(1))}"
            except Exception:
                return m.string

        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)
        return re.sub(r"\b1(s?)\b", r"one\1", s)

    def __call__(self, s: str) -> str:
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word)
        return self.postprocess(s)

class EnglishSpellingNormalizer:
    def __init__(self):
        mapping_path = os.path.join(os.path.dirname(__file__), "english.json")
        with open(mapping_path) as f:
            self.mapping = json.load(f)
    def __call__(self, s: str) -> str:
        return " ".join(self.mapping.get(word, word) for word in (s or "").split())

class EnglishTextNormalizer:
    def __init__(self):
        self.ignore_patterns = re.compile(r"\b(hmm|mm|mhm|mmm|uh|um)\b")
        self.replacers = [(re.compile(p), r) for p, r in {
            r"\bwon't\b": "will not", r"\bcan't\b": "can not", r"\blet's\b": "let us",
            r"\bain't\b": "aint", r"\by'all\b": "you all", r"\bwanna\b": "want to",
            r"\bgotta\b": "got to", r"\bgonna\b": "going to", r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to", r"\bwoulda\b": "would have", r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have", r"\bma'am\b": "madam", r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ", r"\bst\b": "saint ", r"\bdr\b": "doctor ", r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ", r"\bgov\b": "governor ", r"\bald\b": "alderman ", r"\bgen\b": "general ",
            r"\bsen\b": "senator ", r"\brep\b": "representative ", r"\bpres\b": "president ",
            r"\brev\b": "reverend ", r"\bhon\b": "honorable ", r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ", r"\blt\b": "lieutenant ", r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ", r"\bsr\b": "senior ", r"\besq\b": "esquire ",
            r"'d been\b": " had been", r"'s been\b": " has been", r"'d gone\b": " had gone",
            r"'s gone\b": " has gone", r"'d done\b": " had done", r"'s got\b": " has got",
            r"n't\b": " not", r"'re\b": " are", r"'s\b": " is", r"'d\b": " would",
            r"'ll\b": " will", r"'t\b": " not", r"'ve\b": " have", r"'m\b": " am"
        }.items()]
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer()
    def __call__(self, s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        s = re.sub(r"\(([^)]+?)\)", "", s)
        s = self.ignore_patterns.sub("", s)
        s = re.sub(r"\s+'", "'", s)
        for pat, rep in self.replacers:
            s = pat.sub(rep, s)
        s = re.sub(r"(\d),(\d)", r"\1\2", s)
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")
        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)
        return re.sub(r"\s+", " ", s).strip()
