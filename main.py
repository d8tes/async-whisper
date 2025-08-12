import asyncio
import os
import torch
from whisper import load_model, transcribe
from whisper.dub.types.utils import get_writer

async def main():
    model = "tiny"         # or base, small, medium, large
    file = "no.mp3"
    dir = "./outputs"

    os.makedirs(dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = await asyncio.to_thread(load_model, model, device=device)
    result = await transcribe(
        model,
        file,
        temperature=0.0,
        fp16=False
    )
    print(result["text"])
    writer = get_writer("all", dir)
    await asyncio.to_thread(writer, result, file, {})

if __name__ == "__main__":
    asyncio.run(main())
