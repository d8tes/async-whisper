from __future__ import annotations
import sys, re, json, zlib, os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Callable, TextIO, Union
import torch
from torch import Tensor
from numpy import prod

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

system_encoding = sys.getdefaultencoding()

class ResultWriter:
    extension: str
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
    def __call__(self, result: dict, audio_path: str, options: Optional[dict] = None, **kwargs):
        out_path = self.output_dir / f"{Path(audio_path).stem}.{self.extension}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            self.write_result(result, f, options or {}, **kwargs)
    def write_result(self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs):
        raise NotImplementedError

class WriteTXT(ResultWriter):
    extension = "txt"
    def write_result(self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs):
        for seg in result.get("segments") or ():
            t = seg.get("text", "").strip()
            if t:
                file.write(t)
                file.write("\n")

class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str
    def iterate_result(self, result: dict, options: Optional[dict] = None, *,
                       max_line_width: Optional[int] = None,
                       max_line_count: Optional[int] = None,
                       highlight_words: bool = False,
                       max_words_per_line: Optional[int] = None):
        opts = options or {}
        max_line_width = max_line_width or opts.get("max_line_width", 1000)
        max_line_count = max_line_count or opts.get("max_line_count")
        highlight_words = highlight_words or opts.get("highlight_words", False)
        max_words_per_line = max_words_per_line or opts.get("max_words_per_line", 1000)
        preserve_segments = max_line_count is None or max_line_width is None

        def iterate_subtitles():
            line_len, line_count, subtitle = 0, 1, []
            last = (result["segments"][0].get("start") if result.get("segments") else 0.0) or 0.0
            for seg in result.get("segments") or ():
                words = seg.get("words", [])
                wi = 0
                while wi < len(words):
                    wc = min(max_words_per_line, len(words) - wi)
                    for i, orig in enumerate(words[wi:wi + wc]):
                        timing = orig.copy()
                        long_pause = not preserve_segments and timing["start"] - last > 3.0
                        has_room = line_len + len(timing["word"]) <= max_line_width
                        seg_break = i == 0 and subtitle and preserve_segments
                        if line_len > 0 and has_room and not long_pause and not seg_break:
                            line_len += len(timing["word"])
                        else:
                            timing["word"] = timing["word"].strip()
                            if (subtitle and max_line_count is not None and (long_pause or line_count >= max_line_count)) or seg_break:
                                yield subtitle
                                subtitle, line_count = [], 1
                            elif line_len > 0:
                                line_count += 1
                                timing["word"] = "\n" + timing["word"]
                            line_len = len(timing["word"].strip())
                        subtitle.append(timing)
                        last = timing["start"]
                    wi += max_words_per_line
            if subtitle:
                yield subtitle

        if result.get("segments") and "words" in result["segments"][0]:
            for sub in iterate_subtitles():
                sub_start = self.format_timestamp(sub[0]["start"])
                sub_end = self.format_timestamp(sub[-1]["end"])
                sub_text = "".join(w["word"] for w in sub)
                if highlight_words:
                    last = sub_start
                    all_words = [w["word"] for w in sub]
                    for i, this_word in enumerate(sub):
                        s = self.format_timestamp(this_word["start"])
                        e = self.format_timestamp(this_word["end"])
                        if last != s:
                            yield last, s, sub_text
                        yield s, e, "".join(
                            re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", w) if j == i else w 
                            for j, w in enumerate(all_words)
                        )
                        last = e
                else:
                    yield sub_start, sub_end, sub_text
        else:
            for seg in result.get("segments") or ():
                yield (
                    self.format_timestamp(seg["start"]),
                    self.format_timestamp(seg["end"]),
                    seg.get("text", "").strip().replace("-->", "->")
                )

    def format_timestamp(self, seconds: float) -> str:
        return format_timestamp(seconds, always_include_hours=self.always_include_hours, decimal_marker=self.decimal_marker)

class WriteVTT(SubtitlesWriter):
    extension, always_include_hours, decimal_marker = "vtt", False, "."
    def write_result(self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs):
        file.write("WEBVTT\n\n")
        for start, end, text in self.iterate_result(result, options, **kwargs):
            file.write(f"{start} --> {end}\n{text}\n\n")

class WriteSRT(SubtitlesWriter):
    extension, always_include_hours, decimal_marker = "srt", True, ","
    def write_result(self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs):
        for i, (start, end, text) in enumerate(self.iterate_result(result, options, **kwargs), 1):
            file.write(f"{i}\n{start} --> {end}\n{text}\n\n")

class WriteTSV(ResultWriter):
    extension = "tsv"
    def write_result(self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs):
        file.write("start\tend\ttext\n")
        for seg in result.get("segments") or ():
            file.write(f"{round(1000*seg['start'])}\t{round(1000*seg['end'])}\t{seg.get('text','').strip().replace('\t',' ')}\n")

class WriteJSON(ResultWriter):
    extension = "json"
    def write_result(self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs):
        json.dump(result, file, ensure_ascii=False, separators=(",", ":"))

def make_safe(string: str) -> str:
    if system_encoding != "utf-8":
        return string.encode(system_encoding, errors="replace").decode(system_encoding)
    return string

def optional_float(string: str) -> Optional[float]:
    return None if string in ("None", None, "") else float(string)

def optional_int(string: str) -> Optional[int]:
    return None if string in ("None", None, "") else int(string)

def str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"yes", "true", "t", "1"}

if triton:
    @triton.jit
    def dtw_kernel(cost, trace, x, x_stride, cost_stride, trace_stride, N, M, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        for k in range(1, N + M + 1):
            tl.debug_barrier()
            p0, p1, p2 = cost + (k - 1) * cost_stride, cost + k * cost_stride, cost + k * cost_stride + 1
            c0, c1, c2 = tl.load(p0 + offsets, mask=mask), tl.load(p1 + offsets, mask=mask), tl.load(p2 + offsets, mask=mask)
            x_row = tl.load(x + (k - 1) * x_stride + offsets, mask=mask, other=0)
            cost_row = x_row + tl.minimum(tl.minimum(c0, c1), c2)
            tl.store(cost + (k + 1) * cost_stride + 1 + offsets, cost_row, mask=mask)
            tl.store(trace + (k + 1) * trace_stride + 1 + offsets, 2, mask=mask & (c2 <= c0) & (c2 <= c1))
            tl.store(trace + (k + 1) * trace_stride + 1 + offsets, 1, mask=mask & (c1 <= c0) & (c1 <= c2))
            tl.store(trace + (k + 1) * trace_stride + 1 + offsets, 0, mask=mask & (c0 <= c1) & (c0 <= c2))

    @lru_cache(None)
    def median_kernel(filter_width: int):
        @triton.jit
        def kernel(y, x, x_stride, y_stride, BLOCK_SIZE: tl.constexpr):
            row_idx = tl.program_id(0)
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < y_stride
            x_ptr = x + row_idx * x_stride
            y_ptr = y + row_idx * y_stride
            LOAD_ALL_ROWS_HERE
            BUBBLESORT_HERE
            tl.store(y_ptr + offsets, MIDDLE_ROW_HERE, mask=mask)
        kernel = triton.JITFunction(kernel.fn)
        loader = "\n".join([f"    row{i} = tl.load(x_ptr + offsets + {i}, mask=mask)" for i in range(filter_width)])
        bubble = "\n\n".join([
            "\n\n".join([
                "\n".join([
                    f"    smaller = tl.where(row{j} < row{j+1}, row{j}, row{j+1})",
                    f"    larger = tl.where(row{j} > row{j+1}, row{j}, row{j+1})",
                    f"    row{j} = smaller",
                    f"    row{j+1} = larger",
                ]) for j in range(filter_width - i - 1)
            ]) for i in range(filter_width // 2 + 1)
        ])
        src = kernel.src.replace("    LOAD_ALL_ROWS_HERE", loader).replace("    BUBBLESORT_HERE", bubble)
        src = src.replace("MIDDLE_ROW_HERE", f"row{filter_width // 2}")
        if hasattr(kernel, "_unsafe_update_src"):
            kernel._unsafe_update_src(src); kernel.hash = None
        else:
            kernel.src = src
        return kernel

    def median_filter_cuda(x: Tensor, filter_width: int):
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if filter_width <= 0 or filter_width % 2 == 0:
            raise ValueError("filter_width must be a positive odd integer")
        slices = x.contiguous().unfold(-1, filter_width, 1)
        if slices.numel() == 0:
            return x
        grid = int(prod(slices.shape[:-2]))
        kernel = median_kernel(filter_width)
        y = torch.empty_like(slices[..., 0])
        BLOCK_SIZE = 1 << (y.stride(-2) - 1).bit_length()
        kernel[(grid,)](y, x, x.stride(-2), y.stride(-2), BLOCK_SIZE=BLOCK_SIZE)
        return y

def exact_div(x: int, y: int) -> int:
    if y == 0 or x % y != 0:
        raise ValueError(f"{y} must divide {x} evenly")
    return x // y

def compression_ratio(text: str) -> float:
    if not text:
        return 0.0
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data))

def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = ".") -> str:
    if seconds < 0:
        raise ValueError("timestamp must be non-negative")
    ms = round(seconds * 1000.0)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    hours = f"{h:02d}:" if always_include_hours or h > 0 else ""
    return f"{hours}{m:02d}:{s:02d}{decimal_marker}{ms:03d}"

def get_end(segments: List[dict]) -> Optional[float]:
    if not segments:
        return None
    for s in reversed(segments):
        words = s.get("words")
        if words:
            return words[-1]["end"]
    return segments[-1]["end"]

def get_writer(output_format: str, output_dir: Union[str, Path]) -> Callable:
    writers = {"txt": WriteTXT, "vtt": WriteVTT, "srt": WriteSRT, "tsv": WriteTSV, "json": WriteJSON}
    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]
        def write_all(result: dict, file: str, options: Optional[dict] = None, **kwargs):
            for w in all_writers:
                w(result, file, options, **kwargs)
        return write_all
    if output_format not in writers:
        raise ValueError(f"Unsupported output format: {output_format}")
    return writers[output_format](output_dir)
