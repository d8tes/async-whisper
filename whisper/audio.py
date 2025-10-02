import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
CHUNK_LENGTH = 30

N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = (N_SAMPLES + HOP_LENGTH - 1) // HOP_LENGTH
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH
TOKENS_PER_SECOND = SAMPLE_RATE / N_SAMPLES_PER_TOKEN

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

def create_mel_filters(n_fft: int, n_mels: int, sample_rate: int, fmin=0.0, fmax=None) -> torch.Tensor:
    if HAS_LIBROSA:
        fmax = fmax or sample_rate / 2
        mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        return torch.from_numpy(mel_fb).float()
    n_freq = n_fft//2 + 1
    return torch.eye(n_mels, n_freq, dtype=torch.float32)

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
    magnitudes = stft.abs() ** 2  # power spectrogram
    mel_fb = create_mel_filters(N_FFT, n_mels, SAMPLE_RATE).to(magnitudes.device)

    if mel_fb.shape[1] == magnitudes.shape[0]:
        mel_spec = mel_fb @ magnitudes
    elif mel_fb.shape[1] == magnitudes.shape[1]:
        mel_spec = mel_fb @ magnitudes.T
    else:
        raise RuntimeError(f"Shape mismatch mel_fb: {mel_fb.shape} vs magnitudes: {magnitudes.shape}")

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0
