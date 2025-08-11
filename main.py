import asyncio
import os
import sys
import time
import torch
from whisper import load_model
from whisper.dub.transcribe import transcribe
from whisper.dub.types.utils import get_writer

# Edit here for testing
MODEL_NAME = "base"          # base, small, medium, large, turbo
AUDIO_FILE = "no.mp3"        # Path to your audio file
OUTPUT_DIR = "./outputs"     # Where to save outputs
OUTPUT_FORMAT = "all"        # all, txt, vtt, srt, tsv, json

async def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"Audio file not found: {AUDIO_FILE}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading model '{MODEL_NAME}'...")
    t0 = time.time()
    model = await asyncio.to_thread(load_model, MODEL_NAME, device=device)
    print(f"[INFO] Model loaded in {time.time() - t0:.2f}s")

    print(f"[INFO] Starting transcription of {AUDIO_FILE}...")
    t0 = time.time()
    result = await transcribe(
        model,
        AUDIO_FILE,
        temperature=0.0,   # deterministic
        fp16=False,        # force FP32 to avoid CUDA errors on some hardware
        word_timestamps=True,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6
    )
    print(f"[INFO] Transcription finished in {time.time() - t0:.2f}s")
    print("\n===== TRANSCRIPTION TEXT =====\n")
    print(result["text"])
    print("\n==============================\n")

    print(f"[INFO] Writing results to '{OUTPUT_DIR}' in '{OUTPUT_FORMAT}' format...")
    writer = get_writer(OUTPUT_FORMAT, OUTPUT_DIR)
    # Pass writer arguments explicitly; in this case blank, can include highlight_words, etc.
    await asyncio.to_thread(writer, result, AUDIO_FILE, {})

    print("[INFO] Done. Check the outputs directory.")

if __name__ == "__main__":
    asyncio.run(main())
