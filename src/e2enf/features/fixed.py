from logging import getLogger
from typing import Optional

import numpy as np

logger = getLogger(__name__)


def __is_value(y: np.ndarray) -> np.ndarray:
    nonzero_indices = np.nonzero(y)
    vuv = np.zeros_like(y)
    vuv[nonzero_indices] = 1

    return vuv


def get_segment(xs: tuple[np.ndarray], segment: int) -> tuple[np.ndarray]:
    min_len_x = min([x.shape[0] for x in xs])

    limit_idx = min_len_x - segment
    start_idx = np.random.randint(0, limit_idx)
    xs = (x[start_idx : start_idx + segment] for x in xs)

    return xs


def to_continuous(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Convert y to continuous y

    Args:
        y (ndarray): original y sequence with the shape (T)

    Return:
        (ndarray): continuous y with the shape (T)

    """
    # get uv information as binary
    uv = __is_value(y)
    # get start and end of y
    if (y == 0).all():
        logger.warn("all of the y values are 0.")
        return uv, y, False
    start_y = y[y != 0][0]
    end_y = y[y != 0][-1]
    # padding start and end of y sequence
    cy = np.copy(y, "C")
    start_idx = np.where(cy == start_y)[0][0]
    end_idx = np.where(cy == end_y)[0][-1]
    cy[:start_idx] = start_y
    cy[end_idx:] = end_y
    # get non-zero frame index
    nz_frames = np.where(cy != 0)[0]
    # perform linear interpolation
    cy = np.interp(np.arange(0, cy.shape[0]), nz_frames, cy[nz_frames])

    return uv, cy, True


def validate_length(xs: tuple[np.ndarray], ys: Optional[tuple[np.ndarray]] = None, hop_size: int = None):
    """Validate length

    Args:
        xs: numpy array of features
        ys: numpy array of audios
        hop_size: upsampling factor

    Returns:
        length adjusted features
    """
    min_len_x = min([x.shape[0] for x in xs])
    if ys is not None:
        min_len_y = min([y.shape[0] for y in ys])
        if min_len_y < min_len_x * hop_size:
            min_len_x = min_len_y // hop_size
        if min_len_y > min_len_x * hop_size:
            min_len_y = min_len_x * hop_size
        ys = [y[:min_len_y] for y in ys]
    xs = [x[:min_len_x] for x in xs]

    return xs + ys if ys is not None else xs


def adjust_min_len(xs: tuple[np.ndarray]) -> tuple[np.ndarray]:
    """
    fix frame length to minimum length in xs

    Args:
        xs: features need shape (T, ) or (D, T)

    Returns:
        fixed length features
    """
    min_len_x = min([x.shape[-1] for x in xs])
    max_len_x = max([x.shape[-1] for x in xs])
    assert max_len_x - min_len_x <= 3, (
        f"The difference between frames for each feature is too large. {max_len_x - min_len_x} frames"
    )
    xs = [x[:min_len_x] if x.ndim == 1 else x[:, :min_len_x] for x in xs]

    return xs
