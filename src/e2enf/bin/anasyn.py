# A logger for this file
import copy
import os
from logging import getLogger
from os import PathLike
from pathlib import Path

import hydra
import numpy as np
import pyworld as pw
import soundfile as sf
import torch
from hydra.utils import instantiate, to_absolute_path
from joblib import load
from omegaconf import DictConfig
from scipy.interpolate import interp1d

from e2enf.features import (
    f0_utils,
    fixed,
    praat,
    signalgenerator,
    spectral,
    spectrogram,
)
from e2enf.models import dilated_factor
from e2enf.utils import audio_io, file_io, filter, utils

logger = getLogger(__name__)


@torch.no_grad()
@hydra.main(version_base=None, config_path="config", config_name="anasyn")
def main(config: DictConfig) -> None:
    """Run analysis-synthesis process."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Synthesize on {device}.")

    # load pre-trained model from checkpoint file
    model = instantiate(config.generator)
    state_dict = torch.load(to_absolute_path(config.checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict["model"]["generator"])
    logger.info(f"Loaded model parameters from {config.checkpoint_path}.")
    model.remove_weight_norm()
    model.eval().to(device)

    # get scaler
    scaler = load(config.stats)

    # get data processor
    signal_generator = signalgenerator.SignalGenerator(
        sample_rate=config.sample_rate,
        hop_size=config.hop_length,
        sine_amp=config.sine_amp,
        noise_amp=config.noise_amp,
        signal_types=config.signal_types,
    )

    to_stft = spectrogram.STFT(
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        win_size=config.win_size,
        hop_length=config.hop_length,
        fmin=config.fmin,
        fmax=config.fmax,
        clip_val=config.clip_val,
    )
    f0_extractor = f0_utils.F0_Extractor(
        "harvest",
        sample_rate=config.sample_rate,
        hop_size=config.hop_length,
        f0_min=config.minf0,
        f0_max=config.maxf0,
        fix_by_reaper=False,
    )

    # create output directory
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(config.in_dir)
    wav_files = file_io.find_files(in_dir, "*.wav")

    # loop all wav files in in_dir
    for wav_path in wav_files:
        logger.info(f"Start processing {wav_path}")

        wav, sr = audio_io.load_wav(wav_path, target_sr=config.sample_rate)
        if config.highpass_cutoff > 0:
            wav = filter.low_cut_filter(wav, sr, config.highpass_cutoff)

        # Extract F0
        f0 = f0_extractor.extract(wav, return_time=False)
        uv, _cf0, is_all_uv = fixed.to_continuous(f0)
        if is_all_uv:
            lpf_fs = int(config.sample_rate / config.hop_length)
            cf0_lpf = filter.low_pass_filter(_cf0, lpf_fs, cutoff=20)
            next_cutoff = 70
            while not (cf0_lpf >= [0]).all():
                cf0_lpf = filter.low_pass_filter(_cf0, lpf_fs, cutoff=next_cutoff)
                next_cutoff *= 2
        else:
            logger.warning(f"all frame is unvoiced: {wav_path}")
            return None

        # Extract formants
        formants, fo_time = praat.get_formants(
            wav,
            sr=config.sample_rate,
            n_formant=config.n_formant,
            hop_length=config.hop_length,
            win_size=config.fo_win_size,
            fmax=config.fo_max,
            pre_enphasis=config.pre_enphasis,
        )
        formants = praat.fix_formants(formants)

        # Extract spectral centroid and tilt and Energy
        spec = to_stft.get_linear(torch.from_numpy(wav.astype(np.float32)))
        mfbsp = to_stft.to_mel(spec, log=config.log)
        mfbsp = mfbsp.numpy()
        spec = spec.numpy()

        uv, f0, cf0_lpf, spec, mfbsp, formants = fixed.adjust_min_len([uv, f0, cf0_lpf, spec, mfbsp, formants])
        assert uv.shape[0] == f0.shape[0] == cf0_lpf.shape[0] == spec.shape[-1] == mfbsp.shape[-1] == formants.shape[-1]

        mfbsp = mfbsp.T
        centroid = spectral.get_centroid(spec, sr=config.sample_rate, n_fft=config.n_fft).T
        slope = spectral.get_slope(spec)
        energy = spectral.get_energy(spec)
        slope = np.expand_dims(slope, axis=-1)
        energy = np.expand_dims(energy, axis=-1)

        f1, f2, f3, f4 = formants[0], formants[1], formants[2], formants[3]
        _, cf1, _ = fixed.to_continuous(f1)
        _, cf2, _ = fixed.to_continuous(f2)
        _, cf3, _ = fixed.to_continuous(f3)
        _, cf4, _ = fixed.to_continuous(f4)
        uv = np.expand_dims(uv, axis=-1)
        f0 = np.expand_dims(f0, axis=-1)
        f1 = np.expand_dims(f1, axis=-1)
        f2 = np.expand_dims(f2, axis=-1)
        f3 = np.expand_dims(f3, axis=-1)
        f4 = np.expand_dims(f4, axis=-1)
        cf0 = np.expand_dims(cf0_lpf, axis=-1)
        cf1 = np.expand_dims(cf1, axis=-1)
        cf2 = np.expand_dims(cf2, axis=-1)
        cf3 = np.expand_dims(cf3, axis=-1)
        cf4 = np.expand_dims(cf4, axis=-1)

        for f0_factor in config.f0_factors:
            for formants_factor in config.formants_factors:
                # prepare input acoustic features
                c = []
                for feat_type in config.aux_feats:
                    if feat_type == "f0":
                        aux_feat = f0 * f0_factor
                    elif feat_type == "lcf0":
                        aux_feat = np.log(cf0) + np.log(f0_factor)
                    elif feat_type in ["f1", "f2", "f3", "f4", "cf1", "cf2", "cf3", "cf4"]:
                        aux_feat = locals()[feat_type] * formants_factor[int(feat_type[-1]) - 1]
                    else:
                        aux_feat = locals()[feat_type]
                    c += [scaler[f"{feat_type}"].transform(aux_feat)]
                c = np.concatenate(c, axis=1)

                # prepare dense factors
                dfs = []
                for df, us in zip(
                    config.dense_factors,
                    np.cumprod(config.generator.upsample_scales),
                ):
                    dfs += [
                        np.repeat(dilated_factor(cf0, config.sample_rate, df), us)
                        if config.df_f0_type == "cf0"
                        else np.repeat(dilated_factor(f0, config.sample_rate, df), us)
                    ]

                # convert to torch tensors
                f0 = torch.FloatTensor(f0).view(1, 1, -1).to(device)
                cf0 = torch.FloatTensor(cf0).view(1, 1, -1).to(device)
                c = torch.FloatTensor(c).unsqueeze(0).transpose(2, 1).to(device)
                dfs = [torch.FloatTensor(np.array(df)).view(1, 1, -1).to(device) for df in dfs]

                # generate input signals
                if config.sine_f0_type == "cf0":
                    in_signal = signal_generator(cf0)
                elif config.sine_f0_type == "f0":
                    in_signal = signal_generator(f0)

                # synthesize with the neural vocoder
                y = model(in_signal, c, dfs)[0]

                # save output signal as PCM 16 bit wav file
                fo = "_".join([f"{f:.2f}" for f in formants_factor])
                save_path = out_dir.joinpath(Path(wav_path).name.replace(".wav", f"_f{f0_factor:.2f}_fo{fo}.wav"))
                y = y.view(-1).cpu().numpy()
                sf.write(save_path, y, config.sample_rate, "PCM_16")


if __name__ == "__main__":
    main()
