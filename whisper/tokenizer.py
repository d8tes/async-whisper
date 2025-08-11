import base64
import os
import string
import asyncio
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Dict, List, Optional, Tuple
import tiktoken

LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian",
    "ko": "korean", "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish",
    "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish",
    "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech",
    "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian",
    "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian",
    "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian",
    "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian", "mk": "macedonian",
    "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", "ne": "nepali",
    "mn": "mongolian", "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala", "km": "khmer",
    "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans", "oc": "occitan",
    "ka": "georgian", "be": "belarusian", "tg": "tajik", "sd": "sindhi", "gu": "gujarati",
    "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk", "mt": "maltese",
    "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar", "bo": "tibetan", "tl": "tagalog",
    "mg": "malagasy", "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala",
    "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese"
}

TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my", "valencian": "ca", "flemish": "nl", "haitian": "ht", "letzeburgesch": "lb",
    "pushto": "ps", "panjabi": "pa", "moldavian": "ro", "moldovan": "ro", "sinhalese": "si",
    "castilian": "es", "mandarin": "zh"
}

@dataclass
class Tokenizer:
    encoding: tiktoken.Encoding
    num_languages: int
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int] = ()
    special_tokens: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        etoken = self.encoding.encode_single_token
        tokens_set = self.encoding.special_tokens_set
        self.special_tokens = {special: etoken(special) for special in tokens_set}
        sot = self.special_tokens["<|startoftranscript|>"]
        translate = self.special_tokens["<|translate|>"]
        transcribe = self.special_tokens["<|transcribe|>"]
        langs = tuple(LANGUAGES.keys())[:self.num_languages]
        seq = [sot]
        if self.language is not None and self.language in langs:
            seq.append(sot + 1 + langs.index(self.language))
        if self.task is not None:
            seq.append(transcribe if self.task == "transcribe" else translate)
        self.sot_sequence = tuple(seq)

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        filtered = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(filtered, **kwargs)

    def decode_with_timestamps(self, token_ids: List[int], **kwargs) -> str:
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:
        if self.language is None:
            raise ValueError("No language token configured")
        return self.to_language_token(self.language)

    def to_language_token(self, language):
        token = self.special_tokens.get(f"<|{language}|>")
        if token is not None:
            return token
        raise KeyError(f"Language {language} not found")

    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        return tuple(token_id for token, token_id in self.special_tokens.items() if token.strip("<|>") in LANGUAGES)[:self.num_languages]

    @cached_property
    def all_language_codes(self) -> Tuple[str]:
        return tuple(self.decode([tid]).strip("<|>") for tid in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return self.sot_sequence + (self.no_timestamps,)

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += ("<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪").split()
        miscellaneous = set("♩♪♫♬♭♮♯")
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for toks in [self.encoding.encode(symbol), self.encoding.encode(" " + symbol)]:
                if len(toks) == 1 or symbol in miscellaneous:
                    result.add(toks[0])
        return tuple(sorted(result))

    def split_to_word_tokens(self, tokens: List[int]):
        if self.language in {"zh", "ja", "th", "lo", "my", "yue"}:
            return self.split_tokens_on_unicode(tokens)
        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(self, tokens: List[int]):
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"
        words, word_tokens = [], []
        current_tokens = []
        unicode_offset = 0
        append_word = words.append
        append_wtokens = word_tokens.append
        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)
            if replacement_char not in decoded or decoded_full[unicode_offset + decoded.index(replacement_char)] == replacement_char:
                append_word(decoded)
                append_wtokens(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)
        return words, word_tokens

    def split_tokens_on_spaces(self, tokens: List[int]):
        subwords, subword_tokens = self.split_tokens_on_unicode(tokens)
        words, word_tokens = [], []
        append_word = words.append
        append_tokens = word_tokens.append
        for sw, sw_toks in zip(subwords, subword_tokens):
            special = sw_toks[0] >= self.eot
            with_space = sw.startswith(" ")
            punct = sw.strip() in string.punctuation
            if special or with_space or punct or (len(words) == 0):
                append_word(sw)
                append_tokens(sw_toks)
            else:
                words[-1] += sw
                word_tokens[-1].extend(sw_toks)
        return words, word_tokens

@lru_cache(maxsize=None)
def load_encoding_file(vocab_path: str, num_languages: int):
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in open(vocab_path) if line)}
    n_vocab = len(ranks)
    specials = [
        "<|endoftext|>", "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>",
        "<|nospeech|>", "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)]
    ]
    special_tokens = {}
    for i, token in enumerate(specials, start=n_vocab):
        special_tokens[token] = i
    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab + len(specials),
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens
    )

async def get_encoding(name: str = "gpt2", num_languages: int = 99) -> tiktoken.Encoding:
    vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
    return await asyncio.to_thread(load_encoding_file, vocab_path, num_languages)

@lru_cache(maxsize=None)
async def get_tokenizer(multilingual: bool, *, num_languages: int = 99, language: Optional[str] = None, task: Optional[str] = None) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            language = TO_LANGUAGE_CODE.get(language, language)
            if language not in LANGUAGES:
                raise ValueError(f"Unsupported language: {language}")
    encoding_name = "multilingual" if multilingual else "gpt2"
    if multilingual:
        language = language or "en"
        task = task or "transcribe"
    else:
        language = None
        task = None
    encoding = await get_encoding(encoding_name, num_languages)
    return Tokenizer(encoding=encoding, num_languages=num_languages, language=language, task=task)
