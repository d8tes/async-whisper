# Async Whisper - Async Python Whisper Transcription



## Installation

Install from PyPI (if available) or clone and install dependencies manually:

```
pip install async-whisper
```

Or, clone the repo and install:

```
git clone https://github.com/aiosqlite/async-whisper.git
cd async-whisper
pip install -r requirements.txt
python setup.py install
```

> Requires Python 3.8 or higher  
> GPU is recommended but CPU mode is supported

---


```
async-whisper audiofile.mp3 --model base --output_dir ./transcripts --language en --temperature 0.0 --word_timestamps True
```

**arguments:**

- `--model`: whisper model size (tiny, base, small, medium, large, turbo)
- `--language`: language code to transcribe or auto-detect
- `--output_dir`: directory to save output files (all common formats generated)
- `--temperature`: sampling temperature (list or single value)
- `--word_timestamps`: enable detailed word timestamp extraction

---


```
import asyncio
from whisper import load_model, transcribe

async def main():
    model = await load_model("base")  # can specify device with device='cuda'
    result = await transcribe(
        model,
        "audiofile.mp3",
        temperature=0.0,
        fp16=True,
        word_timestamps=True,
    )
    print(result["text"])  # full transcription text
    print(result["segments"])  # list of segments with timestamps and words

asyncio.run(main())
```


```
result = await transcribe(
    model,
    "audiofile.wav",
    language="es",            # spanish language code
    task="translate",         # translate instead of transcribe
    beam_size=5,
    best_of=5,
    patience=None,
    length_penalty=0.6,
    suppress_tokens="-1",
    condition_on_previous_text=True,
    clip_timestamps="0,60",   # transcribe only between 0 and 60 seconds
    word_timestamps=True,
)
```

---

## Configuration Options

| Option                  | Type        | Description                                                       | Default         |
|-------------------------|-------------|-------------------------------------------------------------------|-----------------|
| `model`                 | `str`       | Whisper model size (tiny, base, small, medium, large, turbo)      | `base`          |
| `language`              | `str`       | Language code for transcription or detection                      | Auto-detect     |
| `task`                  | `str`       | Task type: `transcribe` or `translate`                            | `transcribe`    |
| `temperature`           | `float`/list| Sampling temperature(s) for decoding                              | `[0.0,0.2,...]` |
| `beam_size`             | `int`       | Beam width for beam search decoder                                | None            |
| `best_of`               | `int`       | Number of candidate sequences to pick best from                   | None            |
| `patience`              | `float`     | Patience parameter for decoding                                   | None            |
| `length_penalty`        | `float`     | Length penalty for beam search                                    | None            |
| `suppress_tokens`       | `str`/list  | Tokens to suppress during decoding                                | `"-1"`          |
| `condition_on_previous_text` | `bool` | Condition each segment on previous transcription                  | `True`          |
| `clip_timestamps`       | `str`/list  | Comma-separated time ranges for partial transcription             | `"0"`           |
| `word_timestamps`       | `bool`      | Enable word-level timestamps                                      | `False`         |
| `fp16`                  | `bool`      | Use fp16 for faster inference on supported GPUs                   | `True`          |

---

## Features

- Async transcription with `asyncio`
- Support for all Whisper models, from tiny to large/turbo
- Language detection with auto-fallback
- Beam search and greedy decoding, with temperature fallback
- Word-level timestamps aligned with audio
- Multiple output formats: TXT, SRT, VTT, TSV, JSON
- Easy CLI and Python API usage
- GPU mixed precision (fp16) acceleration

---

## Development

```
git clone https://github.com/aiosqlite/async-whisper.git
cd async-whisper
pip install -e .[dev]
pytest
```

```
credits to openai-whispers
support server: [discord.gg/pictures](https://discord.gg/pictures)
```
