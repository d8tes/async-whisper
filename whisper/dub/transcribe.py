import argparse
import os
import traceback
import asyncio
import numpy as np
import torch
import tqdm
from typing import TYPE_CHECKING, Union
from ..audio import FRAMES_PER_SECOND, HOP_LENGTH, N_FRAMES, N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim
from .decoding import DecodingOptions, DecodingResult
from .types.timing import add_word_timestamps
from ..tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .types.utils import exact_div, get_writer, optional_float, optional_int, str2bool

if TYPE_CHECKING:
    from .types.model import Whisper

_tokenizer_cache = {}

async def get_cached_tokenizer(is_multilingual, num_langs, lang, task):
    key = (is_multilingual, num_langs, lang, task)
    if key not in _tokenizer_cache:
        _tokenizer_cache[key] = await get_tokenizer(is_multilingual, num_languages=num_langs, language=lang, task=task)
    return _tokenizer_cache[key]

async def transcribe(model: "Whisper", audio: Union[str, np.ndarray, torch.Tensor], **decode_options) -> dict:
    device = model.device
    fp16 = decode_options.get("fp16", True) and device.type == "cuda"
    if device.type != "cuda":
        fp16 = False
    dtype = torch.float16 if fp16 else torch.float32
    decode_options["fp16"] = fp16

    mel = await asyncio.to_thread(log_mel_spectrogram, audio, model.dims.n_mels, N_SAMPLES)
    content_frames = mel.shape[-1] - N_FRAMES
    mel = mel.to(device=device, dtype=dtype)

    if decode_options.get("language") is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            mel_seg = pad_or_trim(mel, N_FRAMES)
            try:
                _, probs_list = await model.detect_language(mel_seg)
            except Exception:
                probs_list = None
            if isinstance(probs_list, list) and probs_list and isinstance(probs_list[0], dict):
                decode_options["language"] = max(probs_list[0], key=probs_list[0].get)
            else:
                decode_options["language"] = "en"

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = await get_cached_tokenizer(model.is_multilingual, model.num_languages, language, task)

    temperature = decode_options.pop("temperature", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    if not isinstance(temperature, (list, tuple)):
        temperature = [temperature]

    compression_ratio_threshold = decode_options.pop("compression_ratio_threshold", 2.4)
    logprob_threshold = decode_options.pop("logprob_threshold", -1.0)
    no_speech_threshold = decode_options.pop("no_speech_threshold", 0.6)
    clip_timestamps = decode_options.pop("clip_timestamps", "0")
    word_timestamps_flag = decode_options.pop("word_timestamps", False)

    if isinstance(clip_timestamps, str):
        clip_timestamps = [float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])]
    seek_points = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]
    if not seek_points:
        seek_points.append(0)
    if len(seek_points) % 2 == 1:
        seek_points.append(content_frames)
    seek_clips = list(zip(seek_points[::2], seek_points[1::2]))

    input_stride = exact_div(N_FRAMES, model.dims.n_audio_ctx)
    time_precision = input_stride * HOP_LENGTH / SAMPLE_RATE

    all_tokens = []
    all_segments = []
    prompt_reset_since = 0
    remaining_prompt_length = model.dims.n_text_ctx // 2 - 1
    initial_prompt = decode_options.pop("initial_prompt", None)
    carry_initial_prompt = decode_options.pop("carry_initial_prompt", False)
    condition_on_previous_text = decode_options.pop("condition_on_previous_text", True)

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
        remaining_prompt_length -= len(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    decoding_allowed_keys = set(DecodingOptions.__dataclass_fields__.keys())

    async def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
        decode_result = None
        for t in temperature:
            kwargs = {k: v for k, v in decode_options.items() if k in decoding_allowed_keys}
            if t > 0:
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                kwargs.pop("best_of", None)
            options = DecodingOptions(**kwargs, temperature=t)
            try:
                decode_result = await model.decode(segment, options)
            except Exception:
                continue
            needs_fallback = (
                (compression_ratio_threshold is not None and decode_result.compression_ratio > compression_ratio_threshold) or
                (logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold)
            )
            if no_speech_threshold is not None and decode_result.no_speech_prob > no_speech_threshold:
                if not (logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold):
                    needs_fallback = False
            if not needs_fallback:
                return decode_result
        if decode_result is not None:
            return decode_result
        return await model.decode(segment, DecodingOptions(language=language, task=task, temperature=0.0, fp16=fp16))

    clip_idx = 0
    seek = seek_clips[clip_idx][0]

    with tqdm.tqdm(total=content_frames, unit="frames", disable=True) as pbar:
        last_speech_timestamp = 0.0
        while clip_idx < len(seek_clips):
            await asyncio.sleep(0)
            seek_clip_start, seek_clip_end = seek_clips[clip_idx]
            if seek < seek_clip_start:
                seek = seek_clip_start
            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips):
                    seek = seek_clips[clip_idx][0]
                continue

            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment_size = min(N_FRAMES, content_frames - seek, seek_clip_end - seek)
            mel_segment = mel[:, seek: seek + segment_size]
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(device)

            if carry_initial_prompt:
                nignored = max(len(initial_prompt_tokens), prompt_reset_since)
                remaining_prompt = all_tokens[nignored:][-remaining_prompt_length:]
                decode_options["prompt"] = initial_prompt_tokens + remaining_prompt
            else:
                decode_options["prompt"] = all_tokens[prompt_reset_since:]

            result: DecodingResult = await decode_with_fallback(mel_segment)
            tokens = torch.tensor(result.tokens, device=device)
            timestamp_tokens = tokens.ge(tokenizer.timestamp_begin)
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
            consecutive.add_(1)
            current_segments = []

            def new_segment(start: float, end: float, tokens_: torch.Tensor, result_: DecodingResult) -> dict:
                tokens_list = tokens_.tolist()
                text_tokens = [token for token in tokens_list if token < tokenizer.eot]
                return {
                    "seek": seek, "start": start, "end": end, "text": tokenizer.decode(text_tokens),
                    "tokens": tokens_list, "temperature": result_.temperature,
                    "avg_logprob": result_.avg_logprob, "compression_ratio": result_.compression_ratio,
                    "no_speech_prob": result_.no_speech_prob
                }

            if len(consecutive) > 0:
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))
                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_pos = sliced_tokens[0].item() - tokenizer.timestamp_begin
                    end_timestamp_pos = sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    current_segments.append(new_segment(
                        time_offset + start_timestamp_pos * time_precision,
                        time_offset + end_timestamp_pos * time_precision,
                        sliced_tokens, result
                    ))
                    last_slice = current_slice
                seek += segment_size if single_timestamp_ending else (tokens[last_slice - 1].item() - tokenizer.timestamp_begin) * input_stride
            else:
                duration = segment_size * HOP_LENGTH / SAMPLE_RATE
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin:
                    last_timestamp_pos = timestamps[-1].item() - tokenizer.timestamp_begin
                    duration = last_timestamp_pos * time_precision
                current_segments.append(new_segment(time_offset, time_offset + duration, tokens, result))
                seek += segment_size

            if word_timestamps_flag:
                try:
                    await asyncio.to_thread(
                        add_word_timestamps, segments=current_segments, model=model, tokenizer=tokenizer,
                        mel=mel_segment, num_frames=segment_size, last_speech_timestamp=last_speech_timestamp
                    )
                except Exception:
                    pass

            for i, segment in enumerate(current_segments):
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []

            all_segments.extend(
                [{"id": i, **segment} for i, segment in enumerate(current_segments, start=len(all_segments))]
            )
            all_tokens.extend([token for segment in current_segments for token in segment["tokens"]])
            if not condition_on_previous_text or result.temperature > 0.5:
                prompt_reset_since = len(all_tokens)
            pbar.update(min(content_frames, seek) - seek)

    return {
        "text": tokenizer.decode(all_tokens[len(initial_prompt_tokens):]),
        "segments": all_segments,
        "language": language
    }

async def cli_async():
    from .. import available_models, load_model
    def valid_model_name(name):
        if name in available_models() or os.path.exists(name):
            return name
        raise ValueError(f"model should be one of {available_models()} or path to a checkpoint")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str)
    parser.add_argument("--model", default="turbo", type=valid_model_name)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", "-o", type=str, default=".")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["txt","vtt","srt","tsv","json","all"])
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe","translate"])
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]))
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--best_of", type=optional_int, default=5)
    parser.add_argument("--beam_size", type=optional_int, default=5)
    parser.add_argument("--patience", type=float, default=None)
    parser.add_argument("--length_penalty", type=float, default=None)
    parser.add_argument("--suppress_tokens", type=str, default="-1")
    parser.add_argument("--initial_prompt", type=str, default=None)
    parser.add_argument("--carry_initial_prompt", type=str2bool, default=False)
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True)
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2)
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4)
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0)
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6)
    parser.add_argument("--word_timestamps", type=str2bool, default=False)
    parser.add_argument("--prepend_punctuations", type=str, default="\"'“¿([{-")
    parser.add_argument("--append_punctuations", type=str, default="\"'.。,，!！?？:：”)]}、")
    parser.add_argument("--highlight_words", type=str2bool, default=False)
    parser.add_argument("--max_line_width", type=optional_int, default=None)
    parser.add_argument("--max_line_count", type=optional_int, default=None)
    parser.add_argument("--max_words_per_line", type=optional_int, default=None)
    parser.add_argument("--threads", type=optional_int, default=0)
    parser.add_argument("--clip_timestamps", type=str, default="0")
    parser.add_argument("--hallucination_silence_threshold", type=optional_float)
    args = parser.parse_args().__dict__

    model_name = args.pop("model")
    model_dir = args.pop("model_dir")
    output_dir = args.pop("output_dir")
    output_format = args.pop("output_format")
    device = args.pop("device")

    os.makedirs(output_dir, exist_ok=True)
    if model_name.endswith(".en") and args["language"] not in {"en","English"}:
        args["language"] = "en"
    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]
    args["temperature"] = temperature
    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)
    model = await asyncio.to_thread(load_model, model_name, device=device, download_root=model_dir)
    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width", "max_words_per_line"]
    writer_args = {arg: args.pop(arg) for arg in word_options}
    for audio_path in args.pop("audio"):
        try:
            result = await transcribe(model, audio_path, **args)
            await asyncio.to_thread(writer, result, audio_path, **writer_args)
        except Exception:
            traceback.print_exc()

def cli():
    asyncio.run(cli_async())

if __name__ == "__main__":
    cli()
