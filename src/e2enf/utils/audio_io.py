import logging
from os import PathLike

import librosa
import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


def load_wav(full_path, target_sr=None, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)  # than soundfile.
    except Exception as ex:
        logger.warning(f"'{full_path}' failed to load.\nException:")
        logger.warning(ex)
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 48000
        else:
            raise Exception(ex)

    if len(data.shape) > 1:
        data = data[:, 0]
        # check duration of audio file is > 2 samples (because otherwise the slice operation was on the wrong dimension)
        assert len(data) > 2

    if np.issubdtype(data.dtype, np.integer):  # if audio data is type int
        max_mag = -np.iinfo(data.dtype).min  # maximum magnitude = min possible value of intXX
    else:  # if audio data is type fp32
        max_mag = max(np.amax(data), -np.amin(data))
        # data should be either 16-bit INT, 32-bit INT or [-1 to 1] float32
        max_mag = (2**31) + 1 if max_mag > (2**15) else ((2**15) + 1 if max_mag > 1.01 else 1.0)

    data = data.astype(np.float32) / max_mag

    if (np.isinf(data) | np.isnan(data)).any() and return_empty_on_exception:
        # resample will crash with inf/NaN inputs. return_empty_on_exception will return empty arr instead of except
        return [], sampling_rate or target_sr or 48000
    if target_sr is not None and sampling_rate != target_sr:
        data = librosa.core.resample(data, orig_sr=sampling_rate, target_sr=target_sr)
        sampling_rate = target_sr

    return data, sampling_rate


def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)  # than soundfile.
    except Exception as ex:
        logger.warning(f"'{full_path}' failed to load.\nException:")
        logger.warning(ex)
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 48000
        else:
            raise Exception(ex)

    if len(data.shape) > 1:
        data = data[:, 0]
        # check duration of audio file is > 2 samples (because otherwise the slice operation was on the wrong dimension)
        assert len(data) > 2

    if np.issubdtype(data.dtype, np.integer):  # if audio data is type int
        max_mag = -np.iinfo(data.dtype).min  # maximum magnitude = min possible value of intXX
    else:  # if audio data is type fp32
        max_mag = max(np.amax(data), -np.amin(data))
        # data should be either 16-bit INT, 32-bit INT or [-1 to 1] float32
        max_mag = (2**31) + 1 if max_mag > (2**15) else ((2**15) + 1 if max_mag > 1.01 else 1.0)

    data = torch.FloatTensor(data.astype(np.float32)) / max_mag

    # resample will crash with inf/NaN inputs. return_empty_on_exception will return empty arr instead of except
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:
        return [], sampling_rate or target_sr or 48000
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr

    return data, sampling_rate


def save_wav(path: PathLike, wav: np.ndarray, sr: int):
    sf.write(path, wav, sr, "PCM_16")
