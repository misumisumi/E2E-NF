from typing import Optional

import numpy as np
import pyreaper
import pysptk
import pyworld as pw


def sp2mc(sp: np.ndarray, order: int = 24, sr: int = 16000) -> np.ndarray:
    return pysptk.sp2mc(sp, order=order, alpha=pysptk.util.mcepalpha(sr))


def mc2sp(mc: np.ndarray, sr: int = 16000) -> np.ndarray:
    return pysptk.mc2sp(mc, alpha=pysptk.util.mcepalpha(sr))


def code_spenv(sp: np.ndarray, order: int = 24, sr: int = 16000) -> np.ndarray:
    if sp.dtype != np.float64:
        sp = sp.astype(np.float64)

    return pw.code_spectral_envelope(sp, sr, order)


def decode_spenv(mcep: np.ndarray, sr: int = 16000) -> np.ndarray:
    if mcep.dtype != np.float64:
        mcep = mcep.astype(np.float64)

    return pw.decode_spectral_envelope(mcep, sr)


def code_ap(ap: np.ndarray, sr: int = 16000) -> np.ndarray:
    if ap.dtype != np.float64:
        ap = ap.astype(np.float64)

    return pw.code_aperiodicity(ap, sr)


def decode_ap(bap: np.ndarray, sr: int = 16000) -> np.ndarray:
    if bap.dtype != np.float64:
        bap = bap.astype(np.float64)

    return pw.decode_aperiodicity(bap, sr)


def get_sp_envelope(x: np.ndarray, f0: np.ndarray, time: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Calculate spectral envelope using WORLD's cheaptrick."""
    x = x.astype("double", order="C")
    env = pw.cheaptrick(x, f0, time, sample_rate)

    return env.astype(np.float32, order="C")


def get_aperiodicity(x: np.ndarray, f0: np.ndarray, time: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    x = x.astype("double", order="C")
    ap = pw.d4c(x, f0, time, sample_rate)

    return ap.astype(np.float32, order="C")
