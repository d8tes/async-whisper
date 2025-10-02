import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from ..audio import CHUNK_LENGTH
from ..tokenizer import Tokenizer, get_tokenizer
from .types.utils import compression_ratio

if TYPE_CHECKING:
    from .types.model import Whisper

# Global cache to store awaited tokenizer instances by init parameters
_tokenizer_await_cache: Dict[Tuple, Tokenizer] = {}

async def _to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

@torch.no_grad()
async def detect_language(model: "Whisper", mel: Tensor, tokenizer: Optional[Tokenizer] = None) -> Tuple[Tensor, List[dict]]:
    if model is None or mel is None:
        return torch.tensor([]), []
    try:
        if tokenizer is None:
            tokenizer = await get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language=None, task="lang_id")
        if not tokenizer or tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
            return torch.tensor([]), []
        single = mel.ndim == 2
        if single:
            mel = mel.unsqueeze(0)
        if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
            mel = await _to_thread(model.encoder, mel)
        n_audio = mel.shape[0]
        x = torch.full((n_audio, 1), tokenizer.sot, device=mel.device, dtype=torch.long)
        logits = await _to_thread(model.logits, x, mel)
        logits = logits[:, 0]
        mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        mask[list(tokenizer.all_language_tokens)] = False
        logits[:, mask] = float('-inf')
        language_tokens = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1).cpu()
        language_probs = [{c: probs[i, j].item() for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)} for i in range(n_audio)]
        if single:
            language_tokens = language_tokens[0]
            language_probs = language_probs[0]
        return language_tokens, language_probs
    except Exception:
        return torch.tensor([]), []

@dataclass(frozen=True)
class DecodingOptions:
    task: str = "transcribe"
    language: Optional[str] = None
    temperature: float = 0.0
    sample_len: Optional[int] = None
    best_of: Optional[int] = None
    beam_size: Optional[int] = None
    patience: Optional[float] = None
    length_penalty: Optional[float] = None
    prompt: Optional[Union[str, List[int]]] = None
    prefix: Optional[Union[str, List[int]]] = None
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True
    without_timestamps: bool = False
    max_initial_timestamp: Optional[float] = 1.0
    fp16: bool = True

@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan

class Inference:
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor: raise NotImplementedError
    def rearrange_kv_cache(self, source_indices) -> None: raise NotImplementedError
    def cleanup_caching(self) -> None: pass

class PyTorchInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []
        key_modules = [block.attn.key for block in self.model.decoder.blocks]
        value_modules = [block.attn.value for block in self.model.decoder.blocks]
        self.kv_modules = key_modules + value_modules
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        max_ctx = self.model.dims.n_text_ctx
        if tokens.shape[-1] > max_ctx:
            tokens = tokens[:, -max_ctx:]
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()
        if tokens.shape[-1] > self.initial_token_length:
            tokens = tokens[:, -1:]
        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)
    def cleanup_caching(self):
        for hook in self.hooks:
            try:
                hook.remove()
            except:
                pass
        self.kv_cache.clear()
        self.hooks.clear()
    def rearrange_kv_cache(self, source_indices):
        try:
            if source_indices != list(range(len(source_indices))):
                for module in self.kv_modules:
                    self.kv_cache[module] = self.kv_cache[module][source_indices].detach()
        except:
            pass

class MaximumLikelihoodRanker:
    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty
    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        try:
            def scores(logprobs, lengths):
                return [logprob / (length if self.length_penalty is None else ((5 + length) / 6) ** self.length_penalty) for logprob, length in zip(logprobs, lengths)]
            lengths = [[len(t) for t in s] for s in tokens]
            return [np.argmax(scores(p, l)) if len(p) > 0 else 0 for p, l in zip(sum_logprobs, lengths)]
        except:
            return [0] * len(tokens)

class GreedyDecoder:
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot
    def reset(self): pass
    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        next_tokens = logits.argmax(dim=-1) if self.temperature == 0 else Categorical(logits=logits / self.temperature).sample()
        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logits.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)
        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)
        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed
    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()

class BeamSearchDecoder:
    def __init__(self, beam_size: int, eot: int, inference: Inference, patience: Optional[float] = None):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None
        assert self.max_candidates > 0
    def reset(self):
        self.finished_sequences = None
    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")
        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:
            self.finished_sequences = [{} for _ in range(n_audio)]
        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])
                    saved += 1
                    if saved == self.beam_size:
                        break
            finished_sequences.append(finished)
        tokens = torch.tensor(next_tokens, device=tokens.device)
        self.inference.rearrange_kv_cache(source_indices)
        for prev_finished, new_finished in zip(self.finished_sequences, finished_sequences):
            for seq in sorted(new_finished, key=new_finished.get, reverse=True):
                if len(prev_finished) >= self.max_candidates:
                    break
                prev_finished[seq] = new_finished[seq]
        completed = all(len(sequences) >= self.max_candidates for sequences in self.finished_sequences)
        return tokens, completed
    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if len(sequences) < self.beam_size:
                for j in reversed(list(np.argsort(sum_logprobs[i]))):
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break
        tokens = [[torch.tensor(seq) for seq in sequences.keys()] for sequences in self.finished_sequences]
        sum_logprobs = [list(sequences.values()) for sequences in self.finished_sequences]
        return tokens, sum_logprobs

class LogitFilter:
    def apply(self, logits: Tensor, tokens: Tensor) -> None: raise NotImplementedError

class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
    def apply(self, logits: Tensor, tokens: Tensor):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -float('inf')

class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)
    def apply(self, logits: Tensor, tokens: Tensor):
        logits[:, self.suppress_tokens] = -float('inf')

class ApplyTimestampRules(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int, max_initial_timestamp_index: Optional[int]):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index
    def apply(self, logits: Tensor, tokens: Tensor):
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -float('inf')
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin:]
            seq = [t for t in sampled_tokens.tolist()]
            lwts = len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            pwts = len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            if lwts:
                if pwts:
                    logits[k, self.tokenizer.timestamp_begin:] = -float('inf')
                else:
                    logits[k, : self.tokenizer.eot] = -float('inf')
            timestamps = sampled_tokens[sampled_tokens.ge(self.tokenizer.timestamp_begin)]
            if timestamps.numel() > 0:
                tl = timestamps[-1] if (lwts and not pwts) else timestamps[-1] + 1
                logits[k, self.tokenizer.timestamp_begin:tl] = -float('inf')
        if tokens.shape[1] == self.sample_begin:
            logits[:, : self.tokenizer.timestamp_begin] = -float('inf')
            if self.max_initial_timestamp_index is not None:
                last_allowed = self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                logits[:, last_allowed + 1:] = -float('inf')
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            ts_lp = logprobs[k, self.tokenizer.timestamp_begin:].logsumexp(dim=-1)
            max_txt = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if ts_lp > max_txt:
                logits[k, : self.tokenizer.timestamp_begin] = -float('inf')

class DecodingTask:
    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model
        self.tokenizer = None
        self._init_tokenizer_params = dict(
            multilingual=model.is_multilingual,
            num_languages=model.num_languages,
            language=options.language or "en",
            task=options.task
        )
        self.options = self._verify_options(options)

    async def async_init(self):
        key = tuple(sorted(self._init_tokenizer_params.items()))
        global _tokenizer_await_cache
        if key not in _tokenizer_await_cache:
            _tokenizer_await_cache[key] = await get_tokenizer(**self._init_tokenizer_params)
        self.tokenizer = _tokenizer_await_cache[key]

        self.n_group = self.options.beam_size or self.options.best_of or 1
        self.n_ctx = self.model.dims.n_text_ctx
        self.sample_len = self.options.sample_len or self.model.dims.n_text_ctx // 2
        self.sot_sequence = self.tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = self.tokenizer.sot_sequence_including_notimestamps
        self.initial_tokens = self._get_initial_tokens()
        self.sample_begin = len(self.initial_tokens)
        self.sot_index = self.initial_tokens.index(self.tokenizer.sot)
        self.inference = PyTorchInference(self.model, len(self.initial_tokens))
        self.sequence_ranker = MaximumLikelihoodRanker(self.options.length_penalty)
        if self.options.beam_size is not None:
            self.decoder = BeamSearchDecoder(self.options.beam_size, self.tokenizer.eot, self.inference, self.options.patience)
        else:
            self.decoder = GreedyDecoder(self.options.temperature, self.tokenizer.eot)
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not self.options.without_timestamps:
            precision = CHUNK_LENGTH / self.model.dims.n_audio_ctx
            max_initial_timestamp_index = None
            if self.options.max_initial_timestamp:
                max_initial_timestamp_index = round(self.options.max_initial_timestamp / precision)
            self.logit_filters.append(ApplyTimestampRules(self.tokenizer, self.sample_begin, max_initial_timestamp_index))

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError
        if options.temperature == 0 and options.best_of is not None:
            raise ValueError
        if options.patience is not None and options.beam_size is None:
            raise ValueError
        if options.length_penalty is not None and not (0 <= options.length_penalty <= 1):
            raise ValueError
        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)
        if self.options.prefix:
            prefix_tokens = self.tokenizer.encode(" " + self.options.prefix.strip()) if isinstance(self.options.prefix, str) else self.options.prefix
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens += prefix_tokens
        if self.options.prompt:
            prompt_tokens = self.tokenizer.encode(" " + self.options.prompt.strip()) if isinstance(self.options.prompt, str) else self.options.prompt
            tokens = [self.tokenizer.sot_prev] + prompt_tokens[-(self.n_ctx // 2 - 1):] + tokens
        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens
        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]
        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []
        suppress_tokens.extend([
            self.tokenizer.transcribe,
            self.tokenizer.translate,
            self.tokenizer.sot,
            self.tokenizer.sot_prev,
            self.tokenizer.sot_lm,
        ])
        if self.tokenizer.no_speech is not None:
            suppress_tokens.append(self.tokenizer.no_speech)
        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: Tensor):
        if self.options.fp16:
            mel = mel.half()
        if mel.shape[-2:] == (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state):
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)
        if audio_features.dtype != (torch.float16 if self.options.fp16 else torch.float32):
            raise TypeError
        return audio_features

    async def _detect_language(self, audio_features: Tensor, tokens: Tensor):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None
        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = await detect_language(self.model, audio_features, self.tokenizer)
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens
        return languages, lang_probs

    async def _main_loop(self, audio_features: Tensor, tokens: Tensor):
        n_batch = tokens.shape[0]
        sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [float('nan')] * n_batch
        try:
            for i in range(self.sample_len):
                logits = await asyncio.to_thread(self.inference.logits, tokens, audio_features)
                if i == 0 and self.tokenizer.no_speech is not None:
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
                logits = logits[:, -1]
                for filt in self.logit_filters:
                    filt.apply(logits, tokens)
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)
                if completed or tokens.shape[-1] > self.n_ctx:
                    break
                await asyncio.sleep(0)
        finally:
            self.inference.cleanup_caching()
        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    async def run(self, mel: Tensor) -> List[DecodingResult]:
        await self.async_init()
        self.decoder.reset()
        tokenizer = self.tokenizer
        n_audio = mel.shape[0]
        audio_features = await asyncio.to_thread(self._get_audio_features, mel)
        tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)
        languages, language_probs = await self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [DecodingResult(audio_features=feat, language=lang, language_probs=probs) for feat, lang, probs in zip(audio_features, languages, language_probs)]
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)
        tokens, sum_logprobs, no_speech_probs = await self._main_loop(audio_features, tokens)
        audio_features = audio_features[::self.n_group]
        no_speech_probs = no_speech_probs[::self.n_group]
        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens = [[t[self.sample_begin:(t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens]
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts = [tokenizer.decode(t).strip() for t in tokens]
        sum_logprobs = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]
        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError
        return [DecodingResult(audio_features=feat, language=lang, tokens=toks, text=text, avg_logprob=avg_lp, no_speech_prob=no_sp, temperature=self.options.temperature, compression_ratio=compression_ratio(text)) for text, lang, toks, feat, avg_lp, no_sp in zip(*fields)]

@torch.no_grad()
async def decode(model: "Whisper", mel: Tensor, options: DecodingOptions = DecodingOptions(), **kwargs) -> Union[DecodingResult, List[DecodingResult]]:
    if model is None or mel is None:
        return [] if mel is None else DecodingResult(audio_features=torch.empty(0), language="", text="")
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)
    if kwargs:
        options = replace(options, **kwargs)
    task = DecodingTask(model, options)
    results = await task.run(mel)
    if single and results:
        return results[0]
    return results
