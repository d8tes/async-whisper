import numpy as np
import torch
import torch.nn.functional as F
import librosa
from typing import Optional, Union

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 80

CHUNK_LENGTH = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH
N_FRAMES = (N_SAMPLES + HOP_LENGTH - 1) // HOP_LENGTH

FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH
TOKENS_PER_SECOND = SAMPLE_RATE / (HOP_LENGTH * 2)

OVERLAP_SECONDS = 5

def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    waveform, _ = librosa.load(file, sr=sr, mono=True)
    max_amp = np.abs(waveform).max()
    if max_amp > 0:
        waveform = waveform / max_amp
    return waveform.astype(np.float32)

def pad_or_trim(
    array: Union[np.ndarray, torch.Tensor],
    length: int,
    *,
    axis: int = -1,
    pad_mode: str = 'constant',
    center: bool = False,
    constant_value: float = 0.0,
) -> Union[np.ndarray, torch.Tensor]:
    if torch.is_tensor(array):
        current = array.shape[axis]
        if current > length:
            if center:
                start = (current - length) // 2
                indices = torch.arange(start, start + length, device=array.device)
            else:
                indices = torch.arange(length, device=array.device)
            return array.index_select(dim=axis, index=indices)
        elif current < length:
            pad_len = length - current
            pad_size = [0] * (2 * array.ndim)
            pad_size[-2 * axis - 1] = pad_len
            return F.pad(array, pad_size, mode=pad_mode, value=constant_value)
        else:
            return array
    else:
        current = array.shape[axis]
        if current > length:
            if center:
                start = (current - length) // 2
                slices = [slice(None)] * array.ndim
                slices[axis] = slice(start, start + length)
                return array[tuple(slices)]
            else:
                slices = [slice(None)] * array.ndim
                slices[axis] = slice(length)
                return array[tuple(slices)]
        elif current < length:
            pad_width = [(0, 0)] * array.ndim
            pad_width[axis] = (0, length - current)
            return np.pad(array, pad_width, mode=pad_mode, constant_values=constant_value)
        else:
            return array

def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
    padding: int = 0,
    device: Optional[torch.device] = None,
    dither: float = 1e-5,
) -> torch.Tensor:
    if isinstance(audio, str):
        audio = load_audio(audio)
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    if device:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    if dither > 0:
        audio = audio + dither * torch.randn_like(audio)

    window = torch.hann_window(N_FFT, device=audio.device)
    stft_ = torch.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       win_length=N_FFT, window=window,
                       center=True, return_complex=True)
    power_spectrogram = stft_.abs() ** 2

    mel_fb = torch.tensor(librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels),
                          dtype=power_spectrogram.dtype,
                          device=power_spectrogram.device)

    mel_spec = torch.matmul(mel_fb, power_spectrogram)
    mel_spec = torch.clamp(mel_spec, min=1e-10)
    log_mel = torch.log10(mel_spec)
    log_mel = torch.clamp(log_mel, max=log_mel.max())
    normalized = (log_mel + 4.0) / 4.0

    return normalized

def sliding_window_log_mel_spectrogram(
    audio_path: str,
    n_mels: int = N_MELS,
    chunk_seconds: int = CHUNK_LENGTH,
    overlap_seconds: int = OVERLAP_SECONDS,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    audio = load_audio(audio_path)
    sr = SAMPLE_RATE

    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    step = chunk_samples - overlap_samples

    length_audio = len(audio)
    mel_chunks = []

    start = 0
    while start < length_audio:
        end = min(start + chunk_samples, length_audio)
        chunk = audio[start:end]

        if start != 0 and overlap_samples > 0:
            left_ctx_start = max(0, start - overlap_samples)
            left_ctx = audio[left_ctx_start:start]
            chunk = np.concatenate((left_ctx, chunk))

        if len(chunk) < chunk_samples + overlap_samples:
            chunk = np.pad(chunk, (0, chunk_samples + overlap_samples - len(chunk)), mode='constant')

        mel_spec = log_mel_spectrogram(chunk, n_mels=n_mels, device=device)
        mel_chunks.append(mel_spec)

        start += step

    return torch.stack(mel_chunks)
