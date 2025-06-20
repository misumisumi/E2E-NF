import random
from logging import getLogger
from multiprocessing import Manager
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset

from e2enf.features.fixed import validate_length
from e2enf.features.signalgenerator import SignalGenerator
from e2enf.models import dilated_factor
from e2enf.utils.audio_io import load_wav
from e2enf.utils.file_io import check_filename, read_hdf5, read_txt

# A logger for this file
logger = getLogger(__name__)


# TODO: audio_lists->audio_list, feat_lists->feat_list
class AudioFeatDataset(Dataset):
    """PyTorch compatible audio and acoustic feat. dataset."""

    def __init__(
        self,
        audio_lists: List[PathLike] | Dict[int, PathLike],
        feat_lists: List[PathLike] | Dict[int, PathLike],
        stats_lists: List[PathLike] | Dict[int, PathLike],
        shuffle: bool = False,
        audio_length_threshold: Optional[int] = None,
        feat_length_threshold: Optional[int] = None,
        return_filename: bool = False,
        allow_cache: bool = False,
        sample_rate: int = 24000,
        hop_size: int = 120,
        aux_feats: list[str] = [
            "uv",
            "lcf0",
            "cf1",
            "cf2",
            "cf3",
            "cf4",
            "tilt",
            "centroid",
            "energy",
        ],
        f0_factor: float = 1.0,
        formants_factor: list[float] = [1.0, 1.0, 1.0, 1.0],
    ):
        """Initialize dataset.

        Args:
            audio_list: Filename of the list of audio files.
            feat_list: Filename of the list of feature files.
            stats: Filename of the statistic hdf5 file.
            shuffle: Shuffle each file list.
            audio_length_threshold: Threshold to remove short audio files.
            feat_length_threshold: Threshold to remove short feature files.
            return_filename: Whether to return the filename with arrays.
            allow_cache: Whether to allow cache of the loaded files.
            sample_rate: Sampling frequency.
            hop_size: Hope size of acoustic feature
            aux_feats: Type of auxiliary features.
        """
        datasets = []
        # load audio and feature files & check filename
        assert len(audio_lists) == len(feat_lists) == len(stats_lists), (
            f"Number of speaker is different. audio: {len(audio_lists)}, feat: {len(feat_lists)}, stats: {len(stats_lists)}"
        )
        if isinstance(audio_lists, dict):
            spk_ids = list(audio_lists.keys())
        else:
            spk_ids = list(range(len(audio_lists)))

        for spk_id in spk_ids:
            audio_list, feat_list, stats = audio_lists[spk_id], feat_lists[spk_id], stats_lists[spk_id]

            audio_files = read_txt(to_absolute_path(audio_list))
            feat_files = read_txt(to_absolute_path(feat_list))
            assert check_filename(audio_files, feat_files), (
                f"Difference speaker? audio: {audio_files[0]} feat: {feat_files[0]}"
            )
            if audio_length_threshold is not None:
                audio_files, feat_files = self._check_thresholds(
                    audio_files, feat_files, audio_length_threshold, "audio"
                )
            if feat_length_threshold is not None:
                audio_files, feat_files = self._check_thresholds(
                    audio_files, feat_files, feat_length_threshold, "mfbsp"
                )
            assert len(audio_files) != 0, f"{audio_list} is empty."
            assert len(audio_files) == len(feat_files), (
                f"Number of audio and features files are different ({len(audio_files)} vs {len(feat_files)})."
            )
            scaler = joblib.load(to_absolute_path(stats))
            if shuffle:
                # shuffle audio and feature files
                idxs = list(range(len(audio_files)))
                random.shuffle(idxs)
                audio_files = [audio_files[idx] for idx in idxs]
                feat_files = [feat_files[idx] for idx in idxs]
            # flatten the dataset
            for audio_file, feat_file in zip(audio_files, feat_files):
                datasets.append((spk_id, audio_file, feat_file, scaler))

        self.datasets = datasets
        self.spk_ids = spk_ids
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.aux_feats = aux_feats
        self.f0_factor = f0_factor
        self.formants_factor = formants_factor
        logger.info(f"Feature type : {self.aux_feats}")

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(datasets))]

    def _check_thresholds(self, audio_files: list[str], feat_files: list[str], threshold: int, target: str = "audio"):
        """Check if the audio or feature files are longer than the threshold.

        Args:
            audio_files: audio file list.
            feat_files: acoustic feature file list.
            threshold: threshold value.
            target: "audio" or feature name include ".h5"
        """
        if target == "audio":
            lengths = [load_wav(to_absolute_path(f), self.sample_rate).shape[0] for f in audio_files]
        else:
            lengths = [read_hdf5(to_absolute_path(f), f"/{target}").shape[0] for f in feat_files]
        idxs = [idx for idx in range(len(audio_files)) if lengths[idx] > threshold]
        if len(audio_files) != len(idxs):
            logger.warning(f"Some files are filtered by {target} length threshold ({len(audio_files)} -> {len(idxs)}).")
        audio_files = [audio_files[idx] for idx in idxs]
        feat_files = [feat_files[idx] for idx in idxs]

        return audio_files, feat_files

    def _load_sample(self, audio_file: PathLike, feat_file: PathLike, scaler: dict) -> tuple:
        """Load audio and feature from file.

        Args:
            audio_file: Path to the audio file.
            feat_file: Path to the feature file.
            scaler: Path to the scaler dictionary.

        Returns:
            ndarray: Audio signal (T,).
            ndarray: Auxiliary features (T', C).
            ndarray: mel-spectrogram (T', C).
            ndarray: F0 sequence (T', 1).
            ndarray: Continuous F0 sequence (T', 1).¥
            float: F0 mean.
            float: F0 std
        """
        # load audio and features
        audio, sr = load_wav(to_absolute_path(audio_file), self.sample_rate)

        # get auxiliary features
        aux_feats = []
        for feat_type in self.aux_feats:
            if feat_type in ["lcf0"]:
                aux_feat = read_hdf5(to_absolute_path(feat_file), f"/{feat_type.replace('l', '')}")
                aux_feat = np.log(aux_feat) + np.log(self.f0_factor)
            else:
                aux_feat = read_hdf5(to_absolute_path(feat_file), f"/{feat_type}")
                if feat_type in ["cf1", "cf2", "cf3", "cf4"]:
                    aux_feat *= self.formants_factor[int(feat_type[-1]) - 1]
            # scaling auxiliary features
            if feat_type in scaler.keys():
                aux_feat = scaler[f"{feat_type}"].transform(aux_feat)
            aux_feats += [aux_feat]
        aux_feats = np.concatenate(aux_feats, axis=1, dtype=np.float32)

        # get dilated factor sequences
        f0 = read_hdf5(to_absolute_path(feat_file), "/f0")  # discrete F0
        cf0 = read_hdf5(to_absolute_path(feat_file), "/cf0")  # continuous F0

        mfbsp = read_hdf5(to_absolute_path(feat_file), "/mfbsp")  # mel-spectrogram
        mfbsp = scaler["mfbsp"].transform(mfbsp)

        # adjust length
        aux_feats, mfbsp, f0, cf0, audio = validate_length((aux_feats, mfbsp, f0, cf0), (audio,), self.hop_size)
        f0 *= self.f0_factor
        cf0 *= self.f0_factor

        if self.return_filename:
            items = feat_file, audio, aux_feats, mfbsp, f0, cf0
        else:
            items = audio, aux_feats, mfbsp, f0, cf0

        return items

    def __getitem__(self, idx):
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]
        items = self._load_sample(self.datasets[idx][1], self.datasets[idx][2], self.datasets[idx][3])

        if self.allow_cache:
            self.caches[idx] = (self.datasets[idx][0], *items)

        return self.datasets[idx][0], *items  # return spk_id, items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.datasets)


class FeatDataset(Dataset):
    """PyTorch compatible acoustic feat. dataset."""

    def __init__(
        self,
        feat_lists: List[PathLike] | Dict[int, PathLike],
        stats_lists: List[PathLike] | Dict[int, PathLike],
        shuffle: bool = False,
        feat_length_threshold: Optional[int] = None,
        return_filename: bool = False,
        allow_cache: bool = False,
        sample_rate: int = 24000,
        hop_size: int = 120,
        aux_feats: list[str] = [
            "uv",
            "lcf0",
            "cf1",
            "cf2",
            "cf3",
            "cf4",
            "tilt",
            "centroid",
            "energy",
        ],
        f0_factor: float = 1.0,
        formants_factor: list[float] = [1.0, 1.0, 1.0, 1.0],
    ):
        """Initialize dataset.

        Args:
            feat_list: Filename of the list of feature files.
            stats: Filename of the statistic hdf5 file.
            shuffle: Shuffle each file list.
            audio_length_threshold: Threshold to remove short audio files.
            feat_length_threshold: Threshold to remove short feature files.
            return_filename: Whether to return the filename with arrays.
            allow_cache: Whether to allow cache of the loaded files.
            sample_rate: Sampling frequency.
            hop_size: Hope size of acoustic feature
            aux_feats: Type of auxiliary features.
        """
        datasets = []
        # load audio and feature files & check filename
        assert len(feat_lists) == len(stats_lists), (
            f"Number of speaker is different. feat: {len(feat_lists)}, stats: {len(stats_lists)}"
        )
        if isinstance(feat_lists, dict):
            spk_ids = list(feat_lists.keys())
        else:
            spk_ids = list(range(len(feat_lists)))

        for spk_id in spk_ids:
            feat_list, stats = feat_lists[spk_id], stats_lists[spk_id]
            logger.info(stats_lists[spk_id])

            feat_files = read_txt(to_absolute_path(feat_list))
            assert len(feat_files) != 0, f"{feat_list} is empty."
            if feat_length_threshold is not None:
                feat_files = self._check_thresholds(feat_files, feat_length_threshold, "mfbsp")

            scaler = joblib.load(to_absolute_path(stats))
            if shuffle:
                # shuffle audio and feature files
                idxs = list(range(len(feat_files)))
                random.shuffle(idxs)
                feat_files = [feat_files[idx] for idx in idxs]
            # flatten the dataset
            for feat_file in feat_files:
                datasets.append((spk_id, feat_file, scaler))

        self.datasets = datasets
        self.spk_ids = spk_ids
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.aux_feats = aux_feats
        self.f0_factor = f0_factor
        self.formants_factor = formants_factor
        logger.info(f"Feature type : {self.aux_feats}")

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(datasets))]

    def _check_thresholds(self, feat_files: list[str], threshold: int, target: str):
        """Check if the feature files are longer than the threshold.

        Args:
            feat_files: acoustic feature file list.
            threshold: threshold value.
            target: "audio" or feature name include ".h5"
        """
        lengths = [read_hdf5(to_absolute_path(f), f"/{target}").shape[0] for f in feat_files]
        idxs = [idx for idx in range(len(feat_files)) if lengths[idx] > threshold]
        if len(feat_files) != len(idxs):
            logger.warning(f"Some files are filtered by {target} length threshold ({len(feat_files)} -> {len(idxs)}).")
        feat_files = [feat_files[idx] for idx in idxs]

        return feat_files

    def _load_sample(self, feat_file: PathLike, scaler: dict) -> tuple:
        """Load audio and feature from file.

        Args:
            feat_file: Path to the feature file.
            scaler: Path to the scaler dictionary.

        Returns:
            ndarray: Auxiliary features (T', C).
            ndarray: mel-spectrogram (T', C).
            ndarray: F0 sequence (T', 1).
            ndarray: Continuous F0 sequence (T', 1).¥
            float: F0 mean.
            float: F0 std
        """
        # get auxiliary features
        aux_feats = []
        for feat_type in self.aux_feats:
            if feat_type in ["lcf0"]:
                aux_feat = read_hdf5(to_absolute_path(feat_file), f"/{feat_type.replace('l', '')}")
                aux_feat = np.log(aux_feat) + np.log(self.f0_factor)
            else:
                aux_feat = read_hdf5(to_absolute_path(feat_file), f"/{feat_type}")
                if feat_type in ["cf1", "cf2", "cf3", "cf4"]:
                    aux_feat *= self.formants_factor[int(feat_type[-1]) - 1]
            # scaling auxiliary features
            if feat_type in scaler.keys():
                aux_feat = scaler[f"{feat_type}"].transform(aux_feat)
            aux_feats += [aux_feat]
        aux_feats = np.concatenate(aux_feats, axis=1, dtype=np.float32)

        # get dilated factor sequences
        f0 = read_hdf5(to_absolute_path(feat_file), "/f0")  # discrete F0
        cf0 = read_hdf5(to_absolute_path(feat_file), "/cf0")  # continuous F0

        mfbsp = read_hdf5(to_absolute_path(feat_file), "/mfbsp")  # mel-spectrogram
        mfbsp = scaler["mfbsp"].transform(mfbsp)

        # adjust length
        aux_feats, mfbsp, f0, cf0 = validate_length((aux_feats, mfbsp, f0, cf0))

        f0 *= self.f0_factor
        cf0 *= self.f0_factor

        if self.return_filename:
            items = Path(feat_file), aux_feats, mfbsp, f0, cf0
        else:
            items = aux_feats, mfbsp, f0, cf0

        return items

    def __getitem__(self, idx):
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]
        items = self._load_sample(self.datasets[idx][1], self.datasets[idx][2])

        if self.allow_cache:
            self.caches[idx] = (self.datasets[idx][0], *items)

        return self.datasets[idx][0], *items  # return spk_id, items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.datasets)


class SingleCollator(object):
    """Customized collator for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_max_length=12000,
        sample_rate=24000,
        hop_size=120,
        sine_amp=0.1,
        noise_amp=0.003,
        sine_f0_type="cf0",
        signal_types=["sine", "noise"],
        df_f0_type="cf0",
        dense_factors=[0.5, 1, 4, 8],
        upsample_scales=[5, 4, 3, 2],
    ):
        """Initialize customized collator for PyTorch DataLoader.

        Args:
            batch_max_length (int): The maximum length of batch.
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size of auxiliary features.
            sine_amp (float): Amplitude of sine signal.
            noise_amp (float): Amplitude of random noise signal.
            sine_f0_type (str): F0 type for generating sine signal.
            signal_types (list): List of types for input signals.

        """
        if batch_max_length % hop_size != 0:
            batch_max_length += -(batch_max_length % hop_size)
        self.batch_max_length = batch_max_length
        self.batch_max_frames = batch_max_length // hop_size
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.sine_f0_type = sine_f0_type
        self.signal_generator = SignalGenerator(
            sample_rate=sample_rate,
            hop_size=hop_size,
            sine_amp=sine_amp,
            noise_amp=noise_amp,
            signal_types=signal_types,
        )
        self.df_f0_type = df_f0_type
        self.dense_factors = dense_factors
        self.prod_upsample_scales = np.cumprod(upsample_scales)
        self.df_sample_rates = [sample_rate / hop_size * s for s in self.prod_upsample_scales]

    def __call__(self, batch: list):
        """Convert into batch tensors.

        Args:
            batch: list of tuple of the pair of audio and features.

        Returns:
            Tensor: Speaker ID batch (B, 1).
            (Tensor: Gaussian noise (and sine) batch (B, D, T).
            Tensor: Auxiliary feature batch (B, C, T').
            Tensor: F0 sequence batch (B, 1, T').)
            Tensor: Dilated factor batch (B, 1, T).
            Tensor: Target signal batch (B, 1, T).
            Tensor: Mean of F0 batch (B, 1).
            Tensor: Std of F0 batch (B, 1).
        """
        # time resolution check
        # y is the target signal, c is the auxiliary features, f0 is the f0 sequence, cf0 is the cf0 sequence
        spk_batch, y_batch, c_batch, mfbsp_batch, f0_batch, cf0_batch = [], [], [], [], [], []
        dfs_batch = [[] for _ in range(len(self.dense_factors))]
        for idx in range(len(batch)):
            # spk_id, audio, aux_feats, f0, cf0
            spk_id, x, c, mfbsp, f0, cf0 = batch[idx]
            if len(c) > self.batch_max_frames:
                # randomly pickup with the batch_max_length length of the part
                start_frame = np.random.randint(0, len(c) - self.batch_max_frames)
                start_step = start_frame * self.hop_size
                y = x[start_step : start_step + self.batch_max_length]
                c = c[start_frame : start_frame + self.batch_max_frames]
                mfbsp = mfbsp[start_frame : start_frame + self.batch_max_frames]
                f0 = f0[start_frame : start_frame + self.batch_max_frames]
                cf0 = cf0[start_frame : start_frame + self.batch_max_frames]
                dfs = []
                for df, us in zip(self.dense_factors, self.prod_upsample_scales):
                    dfs += [
                        np.repeat(dilated_factor(cf0, self.sample_rate, df), us)
                        if self.df_f0_type == "cf0"
                        else np.repeat(dilated_factor(f0, self.sample_rate, df), us)
                    ]
                self._check_length(y, c, mfbsp, f0, cf0, dfs)
            else:
                logger.warn(f"Removed short sample from batch (length={len(x)}).")
                continue
            spk_batch += [spk_id]  # [(1,), ...]
            y_batch += [y.astype(np.float32).reshape(-1, 1)]  # [(T, 1), ...]
            c_batch += [c.astype(np.float32)]  # [(T', D), ...]
            for i in range(len(self.dense_factors)):
                dfs_batch[i] += [dfs[i].astype(np.float32).reshape(-1, 1)]  # [(T', 1), ...]
            mfbsp_batch += [mfbsp.astype(np.float32)]
            f0_batch += [f0.astype(np.float32).reshape(-1, 1)]  # [(T', 1), ...]
            cf0_batch += [cf0.astype(np.float32).reshape(-1, 1)]  # [(T', 1), ...]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = torch.FloatTensor(np.array(y_batch)).transpose(2, 1)  # (B, 1, T)
        c_batch = torch.FloatTensor(np.array(c_batch)).transpose(2, 1)  # (B, D, T')
        mfbsp_batch = torch.FloatTensor(np.array(mfbsp_batch)).transpose(2, 1)  # (B, D, T')
        for i in range(len(self.dense_factors)):
            dfs_batch[i] = torch.FloatTensor(np.array(dfs_batch[i])).transpose(2, 1)  # (B, 1, T')
        f0_batch = torch.FloatTensor(np.array(f0_batch)).transpose(2, 1)  # (B, 1, T')
        cf0_batch = torch.FloatTensor(np.array(cf0_batch)).transpose(2, 1)  # (B, 1, T')

        # make input signal batch tensor
        if self.sine_f0_type == "cf0":
            in_batch = self.signal_generator(cf0_batch)
        elif self.sine_f0_type == "f0":
            in_batch = self.signal_generator(f0_batch)
        else:
            in_batch = None

        return spk_batch, (in_batch, c_batch, f0_batch), dfs_batch, y_batch, mfbsp_batch

    def _check_length(self, x, c, mfbsp, f0, cf0, dfs):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == len(c) * self.hop_size
        assert len(x) == len(mfbsp) * self.hop_size
        assert len(x) == len(f0) * self.hop_size
        assert len(x) == len(cf0) * self.hop_size
        for i in range(len(self.dense_factors)):
            assert len(x) * self.df_sample_rates[i] == len(dfs[i]) * self.sample_rate


class SingleFMCollator(object):
    """Customized collator for Pytorch DataLoader in training."""

    def __init__(self, batch_max_frames=46):
        """Initialize customized collator for PyTorch DataLoader.

        Args:
            batch_max_frame (int): The maximum frame of batch.
        """
        self.batch_max_frames = batch_max_frames

    def __call__(self, batch: list):
        """Convert into batch tensors.

        Args:
            batch: list of tuple of the pair of audio and features.

        Returns:
            Tensor: Speaker ID batch (B, 1).
            Tensor: Auxiliary feature batch (B, C, T').
            Tensor: Target mel-spectrogram batch (B, D, T).
        """
        # time resolution check
        # c is the auxiliary features, f0 is the f0 sequence, cf0 is the cf0 sequence
        spk_batch, c_batch, mfbsp_batch = [], [], []
        for idx in range(len(batch)):
            # spk_id, audio, aux_feats, f0, cf0
            spk_id, c, mfbsp, f0, cf0 = batch[idx]
            if len(c) > self.batch_max_frames:
                # randomly pickup with the batch_max_length length of the part
                start_frame = np.random.randint(0, len(c) - self.batch_max_frames)
                c = c[start_frame : start_frame + self.batch_max_frames]
                mfbsp = mfbsp[start_frame : start_frame + self.batch_max_frames]
                assert len(c) == len(mfbsp)
            else:
                logger.warn(f"Removed short sample from batch (length={len(c)}).")
                continue
            spk_batch += [spk_id]  # [(1,), ...]
            c_batch += [c.astype(np.float32)]  # [(T', D), ...]
            mfbsp_batch += [mfbsp.astype(np.float32)]

        # convert each batch to tensor, asuume that each item in batch has the same length
        c_batch = torch.FloatTensor(np.array(c_batch)).transpose(2, 1)  # (B, D, T')
        mfbsp_batch = torch.FloatTensor(np.array(mfbsp_batch)).transpose(2, 1)  # (B, D, T')

        return spk_batch, c_batch, mfbsp_batch

    def _check_length(self, x, c, mfbsp, f0, cf0, dfs):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == len(c) * self.hop_size
        assert len(x) == len(mfbsp) * self.hop_size
        assert len(x) == len(f0) * self.hop_size
        assert len(x) == len(cf0) * self.hop_size
        for i in range(len(self.dense_factors)):
            assert len(x) * self.df_sample_rates[i] == len(dfs[i]) * self.sample_rate
