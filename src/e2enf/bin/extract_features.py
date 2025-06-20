import os
from logging import getLogger
from os import PathLike
from pathlib import Path

import hydra
import numpy as np
import pyworld as pw
import torch
from hydra.utils import to_absolute_path
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from e2enf.features import f0_utils, fixed, praat, spectral, spectrogram, world
from e2enf.utils import audio_io, file_io, filter, utils

# TODO: logging in multiprocessing
logger = getLogger(__name__)


def path_create(file_list: list[PathLike], inputpath: PathLike, outputpath: PathLike, ext: str = None):
    for filepath in file_list:
        path_replace(filepath, inputpath, outputpath, ext)


def path_replace(filepath: PathLike, inputpath: PathLike, outputpath: PathLike, ext: str = None) -> Path:
    filepath = str.replace(str(filepath), str(inputpath), str(outputpath))
    filepath = Path(filepath)
    if ext is not None:
        filepath = filepath.with_suffix(ext)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    return filepath


def aux_list_create(list_file: PathLike, config: DictConfig):
    """Create list of auxiliary acoustic features

    Args:
        list_file (str): Filename of wav list
        config (dict): Config

    """
    aux_list_file = list_file.replace(".scp", ".list")
    wav_files = file_io.read_txt(list_file)
    with open(aux_list_file, "w") as f:
        for wav_name in wav_files:
            feat_name = path_replace(
                wav_name,
                config.in_dir,
                config.out_dir,
                ext=config.feature_format,
            )
            f.write(f"{feat_name}\n")


def process(filepath: PathLike, config: DictConfig):
    logger = getLogger(__name__)
    wav, sr = audio_io.load_wav(filepath, target_sr=config.sample_rate)
    if (wav == 0).all():
        logger.warning(f"missing file: {filepath}")

        return None
    if config.highpass_cutoff > 0:
        wav = filter.low_cut_filter(wav, sr, config.highpass_cutoff)
    to_stft = spectrogram.STFT(
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        win_size=config.win_size,
        hop_length=config.hop_size,
        fmin=config.fmin,
        fmax=config.fmax,
        clip_val=config.clip_val,
    )
    f0_extractor = f0_utils.F0_Extractor(
        "harvest",
        sample_rate=config.sample_rate,
        hop_size=config.hop_size,
        f0_min=config.minf0,
        f0_max=config.maxf0,
        fix_by_reaper=False,
    )

    spec = to_stft.get_linear(torch.from_numpy(wav.astype(np.float32)))
    mfbsp = to_stft.to_mel(spec, log=config.log)
    mfbsp = mfbsp.numpy()

    f0, time = f0_extractor.extract(wav, return_time=True)
    uv, cf0, is_all_uv = fixed.to_continuous(f0)
    if is_all_uv:
        lpf_fs = int(config.sample_rate / config.hop_size)
        cf0_lpf = filter.low_pass_filter(cf0, lpf_fs, cutoff=20)
        next_cutoff = 70
        while not (cf0_lpf >= [0]).all():
            cf0_lpf = filter.low_pass_filter(cf0, lpf_fs, cutoff=next_cutoff)
            next_cutoff *= 2
    else:
        logger.warning(f"all frame is unvoiced: {filepath}")
        return None

    env = world.get_sp_envelope(wav, f0, time, sample_rate=config.sample_rate)
    mcep = world.sp2mc(env, order=config.mcep_order, sr=config.sample_rate).T
    ap = world.get_aperiodicity(wav, f0, time, sample_rate=config.sample_rate)
    bap = world.code_ap(ap, sr=config.sample_rate).T

    formants, fo_time = praat.get_formants(
        wav,
        sr=config.sample_rate,
        n_formant=config.n_formant,
        hop_length=config.hop_size,
        win_size=config.fo_win_size,
        fmax=config.fo_max,
        pre_enphasis=config.pre_enphasis,
    )
    formants = praat.fix_formants(formants)

    uv, f0, cf0_lpf, spec, mfbsp, formants, mcep, bap = fixed.adjust_min_len(
        [uv, f0, cf0_lpf, spec, mfbsp, formants, mcep, bap]
    )
    assert (
        uv.shape[0]
        == f0.shape[0]
        == cf0_lpf.shape[0]
        == spec.shape[-1]
        == mfbsp.shape[-1]
        == formants.shape[-1]
        == mcep.shape[-1]
        == bap.shape[-1]
    ), (
        f"shape mismatch: uv {uv.shape}, f0 {f0.shape}, cf0_lpf {cf0_lpf.shape}, spec {spec.shape}, mfbsp {mfbsp.shape}, formants {formants.shape}, mcep {mcep.shape}, bap {bap.shape}"
    )

    f1, f2, f3, f4 = formants[0], formants[1], formants[2], formants[3]
    _, cf1, _ = fixed.to_continuous(f1)
    _, cf2, _ = fixed.to_continuous(f2)
    _, cf3, _ = fixed.to_continuous(f3)
    _, cf4, _ = fixed.to_continuous(f4)

    spec = spec.numpy()
    centroid = spectral.get_centroid(spec, sr=config.sample_rate, n_fft=config.n_fft)
    slope = spectral.get_slope(spec)
    energy = spectral.get_energy(spec)

    # expand to (T, 1)
    uv = np.expand_dims(uv, axis=-1)
    f0 = np.expand_dims(f0, axis=-1)
    f1 = np.expand_dims(f1, axis=-1)
    f2 = np.expand_dims(f2, axis=-1)
    f3 = np.expand_dims(f3, axis=-1)
    f4 = np.expand_dims(f4, axis=-1)
    cf0_lpf = np.expand_dims(cf0_lpf, axis=-1)
    cf1 = np.expand_dims(cf1, axis=-1)
    cf2 = np.expand_dims(cf2, axis=-1)
    cf3 = np.expand_dims(cf3, axis=-1)
    cf4 = np.expand_dims(cf4, axis=-1)
    slope = np.expand_dims(slope, axis=-1)
    energy = np.expand_dims(energy, axis=-1)

    feat_name = path_replace(
        filepath,
        config.in_dir,
        config.out_dir,
        ext=config.feature_format,
    )
    feat_name = to_absolute_path(feat_name)

    # feature shape (T, 1) or (T, D)
    file_io.write_hdf5(feat_name, "/uv", uv)
    file_io.write_hdf5(feat_name, "/f0", f0)
    file_io.write_hdf5(feat_name, "/f1", f1)
    file_io.write_hdf5(feat_name, "/f2", f2)
    file_io.write_hdf5(feat_name, "/f3", f3)
    file_io.write_hdf5(feat_name, "/f4", f4)
    file_io.write_hdf5(feat_name, "/cf0", cf0_lpf)
    file_io.write_hdf5(feat_name, "/cf1", cf1)
    file_io.write_hdf5(feat_name, "/cf2", cf2)
    file_io.write_hdf5(feat_name, "/cf3", cf3)
    file_io.write_hdf5(feat_name, "/cf4", cf4)
    file_io.write_hdf5(feat_name, "/slope", slope)
    file_io.write_hdf5(feat_name, "/energy", energy)
    file_io.write_hdf5(feat_name, "/centroid", centroid.T)
    file_io.write_hdf5(feat_name, "/mfbsp", mfbsp.T)
    file_io.write_hdf5(feat_name, "/mcep", mcep.T)
    file_io.write_hdf5(feat_name, "/bap", bap.T)


@hydra.main(version_base=None, config_path="config", config_name="extract_features")
def main(config: DictConfig):
    # show default argument
    logger.info(OmegaConf.to_yaml(config))

    # read file list
    file_list = file_io.read_txt(to_absolute_path(config.file_list))
    logger.info(f"number of atterances = {len(file_list)}")

    # list division
    if config.spkinfo and Path(to_absolute_path(config.spkinfo)).exists():
        # load speaker info
        with open(to_absolute_path(config.spkinfo), "r") as f:
            spkinfo = OmegaConf.load(f)
        logger.info(f"Spkinfo {config.spkinfo} is used.")
        # divide into each spk list
        file_and_conf_list = utils.spk_division(file_list, config, spkinfo)
    else:
        logger.info(f"Since spkinfo {config.spkinfo} is not exist, default f0 range and power threshold are used.")
        file_and_conf_list = [(file, config) for file in file_list]

    # set mode
    if config.inv:
        # create auxiliary feature list
        aux_list_create(to_absolute_path(config.file_list), config)
        # create folder
        path_create(file_list, config.in_dir, config.out_dir, config.feature_format)

    _ = [
        r
        for r in tqdm(
            Parallel(n_jobs=config.n_jobs, return_as="generator")(
                (delayed(process)(path, config) for path, config in file_and_conf_list)
            ),
            total=len(file_list),
        )
    ]


if __name__ == "__main__":
    main()
