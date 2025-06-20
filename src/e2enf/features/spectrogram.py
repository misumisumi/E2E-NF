import os
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from torch import Tensor

logger = getLogger(__name__)
os.environ["LRU_CACHE_CAPACITY"] = "3"


def dynamic_range_compression(x: np.array, C: float = 1.0, clip_val: float = 1e-5, log="log"):
    return getattr(np, log)(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x: np.array, C: float = 1.0, log="log"):
    if log == "log":
        return np.exp(x) / C
    elif log == "log2":
        return np.power(2, x) / C
    elif log == "log10":
        return np.power(10, x) / C
    else:
        error_msg = f"Invalid log type: {log}"
        ValueError(error_msg)


def dynamic_range_compression_torch(x: Tensor, C: float = 1.0, clip_val: float = 1e-5, log="log"):
    return getattr(torch, log)(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x: Tensor, C: float = 1.0, log="log"):
    if log == "log":
        return torch.exp(x) / C
    elif log == "log2":
        return torch.pow(2, x) / C
    elif log == "log10":
        return torch.pow(10, x) / C
    else:
        error_msg = f"Invalid log type: {log}"
        ValueError(error_msg)


class STFT:
    def __init__(
        self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-5
    ):
        self.target_sr = sr

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    @torch.no_grad()
    def get_linear(self, y, keyshift=0, speed=1, center=False) -> Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(0)
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length

        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))

        keyshift_key = str(keyshift) + "_" + str(y.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)

        pad_left = (n_fft_new - hop_length_new) // 2
        pad_right = max((n_fft_new - hop_length_new + 1) // 2, n_fft_new - y.size(-1) - pad_left)
        if pad_right < y.size(-1):
            mode = "reflect"
        else:
            mode = "constant"
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode)
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft_new,
            hop_length=hop_length_new,
            win_length=win_size_new,
            window=self.hann_window[keyshift_key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + (1e-9))
        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * win_size / win_size_new

        return spec.squeeze(0)

    @torch.no_grad()
    def to_mel(self, spec: Tensor, log: str = "log") -> Tensor:
        """return log-scale mel spctrogram

        Args:
            spec: linear spectrogram
            log: log-scale

        Returns:
            (log-scale) mel spectrogram
        """
        sampling_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val
        mel_basis_key = str(fmax) + "_" + str(spec.device)

        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(spec.device)

        mfbsp = torch.matmul(self.mel_basis[mel_basis_key], spec)
        if log is not None:
            mfbsp = dynamic_range_compression_torch(mfbsp, clip_val=clip_val, log=log)

        return mfbsp

    def __call__(self, audio: Tensor) -> Tensor:
        spect = self.get_linear(audio)
        mfbsp = self.to_mel(spect)

        return mfbsp
