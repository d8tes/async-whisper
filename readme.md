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


## configuration options

| option                   | type        | description                                                     | default          |
|--------------------------|-------------|-----------------------------------------------------------------|------------------|
| `model`                  | `str`       | whisper model size (tiny, base, small, medium, large, turbo)    | `base`           |
| `language`               | `str`       | language code for transcription or detection                    | auto-detect      |
| `task`                   | `str`       | task type: `transcribe` or `translate`                          | `transcribe`     |
| `temperature`            | `float`/list| sampling temperature(s) for decoding                            | `[0.0,0.2,...]`  |
| `beam_size`              | `int`       | beam width for beam search decoder                              | None             |
| `best_of`                | `int`       | number of candidate sequences to pick best from                 | None             |
| `patience`               | `float`     | patience parameter for decoding                                 | None             |
| `length_penalty`         | `float`     | length penalty for beam search                                  | None             |
| `suppress_tokens`        | `str`/list  | tokens to suppress during decoding                              | `"-1"`           |
| `condition_on_previous_text` | `bool`  | condition each segment on previous transcription                | `True`           |
| `clip_timestamps`        | `str`/list  | comma-separated time ranges for partial transcription           | `"0"`            |
| `word_timestamps`        | `bool`      | enable word-level timestamps                                    | `False`          |
| `fp16`                   | `bool`      | use fp16 for faster inference on supported gpus                 | `True`           |

---

## features

- async transcription with `asyncio`  
- support for all whisper models, from tiny to large/turbo  
- language detection with auto-fallback  
- beam search and greedy decoding, with temperature fallback  
- word-level timestamps aligned with audio  
- multiple output formats: txt, srt, vtt, tsv, json  
- easy cli and python api usage  
- gpu mixed precision (fp16) acceleration  

---

## development


## Development

```
git clone https://github.com/aiosqlite/async-whisper.git
cd async-whisper
pip install -e .[dev]
pytest
```

## Credits
```
credits to openai-whispers, https://github.com/openai/whisper/blob/main/requirements.txt  
support server: [discord.gg/pictures](https://discord.gg/pictures)  
created by keron 
```


---

## License
This project is licensed under the [MIT License](LICENSE).
