import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

from .dub.types.utils import exact_div

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)

@lru_cache(maxsize=64)
def _cached_load_audio(path: str, sr: int) -> np.ndarray:
    return _load_audio(path, sr)

def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    return _cached_load_audio(file, sr)

def _load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    if HAS_TORCHAUDIO:
        try:
            waveform, loaded_sr = torchaudio.load(file)
            if loaded_sr != sr:
                waveform = torchaudio.functional.resample(waveform, loaded_sr, sr)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
            return waveform.numpy().astype(np.float32)
        except Exception:
            pass
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", file,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio {file}: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

def pad_or_trim(
    array: Union[np.ndarray, torch.Tensor],
    length: int = N_SAMPLES,
    *,
    axis: int = -1,
    pad_mode: str = 'constant',
    center: bool = False,
    constant_value: float = 0.0
) -> Union[np.ndarray, torch.Tensor]:

    if torch.is_tensor(array):
        current = array.shape[axis]
        if current > length:
            if center:
                start = (current - length) // 2
                indices = torch.arange(start, start + length, device=array.device)
            else:
                indices = torch.arange(length, device=array.device)
            array = array.index_select(dim=axis, index=indices)
        elif current < length:
            pad_len = length - current
            if pad_mode == 'constant':
                pad_width = [(0, 0)] * array.ndim
                pad_width[axis] = (0, pad_len)
                array = F.pad(array, [p for extents in reversed(pad_width) for p in extents], value=constant_value)
            elif pad_mode == 'reflect':
                array = F.pad(array, (0, pad_len), mode='reflect')
            else:
                raise ValueError(f"Unsupported pad_mode {pad_mode} for tensors")
        return array
    else:
        current = array.shape[axis]
        if current > length:
            if center:
                start = (current - length) // 2
                slices = [slice(None)] * array.ndim
                slices[axis] = slice(start, start + length)
                array = array[tuple(slices)]
            else:
                slices = [slice(None)] * array.ndim
                slices[axis] = slice(length)
                array = array[tuple(slices)]
        elif current < length:
            pad_width = [(0, 0)] * array.ndim
            pad_width[axis] = (0, length - current)
            array = np.pad(array, pad_width=pad_width, mode='constant', constant_values=constant_value)
        return array

@lru_cache(maxsize=None)
def get_hann_window(device: Union[str, torch.device], n_fft: int = N_FFT):
    return torch.hann_window(n_fft, dtype=torch.float32, device=device)

@lru_cache(maxsize=None)
def mel_filters(device: Union[str, torch.device], n_mels: int, n_fft: int = N_FFT):
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, mmap_mode="r", allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

@lru_cache(maxsize=64)
def get_mel_transform(
    device: Union[str, torch.device],
    n_mels: int,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
    f_min: float = 0.0,
    f_max: Optional[float] = None
):
    if not HAS_TORCHAUDIO:
        return None
    f_max = f_max or sample_rate / 2
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window_fn=torch.hann_window,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        mel_scale='htk',
        f_min=f_min,
        f_max=f_max
    ).to(device)
    return mel_spec

def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    dither: float = 1e-5
) -> torch.Tensor:
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    if dither > 0:
        audio = audio + dither * torch.randn_like(audio)
    mel_transform = get_mel_transform(audio.device if device is None else device, n_mels, n_fft=N_FFT, hop_length=HOP_LENGTH)
    if mel_transform is not None:
        mel_spec = mel_transform(audio)
    else:
        window = get_hann_window(audio.device if device is None else device, N_FFT)
        stft = torch.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        filters = mel_filters(audio.device if device is None else device, n_mels)
        mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0
