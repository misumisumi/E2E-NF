import os
import re
from logging import getLogger
from pathlib import Path
from time import time

import hydra
import numpy as np
import soundfile as sf
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from e2enf.datasets import FeatDataset
from e2enf.features.signalgenerator import SignalGenerator
from e2enf.models import dilated_factor

# A logger for this file
logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="decode_fm")
def main(config: DictConfig) -> None:
    """Run decoding process."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # set device
    if config.device != "":
        device = torch.device(config.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Decode on {device}.")

    # load pre-trained model from checkpoint file
    if config.checkpoint_path is None:
        checkpoint_steps = config.checkpoint_steps
        checkpoint_path = os.path.join(
            config.out_dir,
            "checkpoints",
            f"checkpoint-{config.checkpoint_steps}steps.pkl",
        )
    else:
        pattern = re.compile(r"\d+")
        m = pattern.search(config.checkpoint_path)
        checkpoint_steps = m.group() if m is not None else "unknown"
        checkpoint_path = config.checkpoint_path
    # check directory existence
    out_dir = Path(to_absolute_path(config.out_dir))
    out_dir = out_dir.joinpath("wav", str(checkpoint_steps))

    state_dict = torch.load(to_absolute_path(checkpoint_path), map_location="cpu", weights_only=False)
    logger.info(f"Loaded model parameters from {checkpoint_path}.")
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(state_dict["model"]["feature-mapping"])
    model.remove_weight_norm()
    model.eval().to(device)

    vocoder_state_dict = torch.load(
        to_absolute_path(config.vocoder_checkpoint_path), map_location="cpu", weights_only=False
    )
    vocoder = hydra.utils.instantiate(config.vocoder)
    vocoder.load_state_dict(vocoder_state_dict["model"]["generator"])
    vocoder.remove_weight_norm()
    vocoder.eval().to(device)

    for f0_factor in config.f0_factors:
        for formants_factor in config.formants_factors:
            dataset = FeatDataset(
                feat_lists=config.data.eval_feat,
                stats_lists=config.data.stats,
                allow_cache=config.data.allow_cache,
                sample_rate=config.data.sample_rate,
                hop_size=config.data.hop_size,
                aux_feats=config.data.aux_feats,
                f0_factor=f0_factor,
                formants_factor=formants_factor,
                return_filename=True,
            )
            logger.info(f"The number of features to be decoded = {len(dataset)}.")

            signal_generator = SignalGenerator(
                sample_rate=config.data.sample_rate,
                hop_size=config.data.hop_size,
                sine_amp=config.data.sine_amp,
                noise_amp=config.data.noise_amp,
                signal_types=config.data.signal_types,
            )

            with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
                total_rtf = 0.0
                for idx, items in enumerate(pbar, 1):
                    _, feat_path, c_x, mfbsp_x, f0, cf0 = items
                    if config.mfbsp2spkparams:
                        c_x = mfbsp_x
                    # create dense factors
                    dfs = []
                    for df, us in zip(
                        config.data.dense_factors,
                        np.cumprod(config.vocoder.upsample_scales),
                    ):
                        dfs += [
                            np.repeat(dilated_factor(cf0, config.data.sample_rate, df), us)
                            if config.data.df_f0_type == "cf0"
                            else np.repeat(dilated_factor(f0, config.data.sample_rate, df), us)
                        ]
                    c_x = torch.FloatTensor(c_x).unsqueeze(0).transpose(2, 1).to(device)
                    f0 = torch.FloatTensor(f0).view(1, 1, -1).to(device)
                    cf0 = torch.FloatTensor(cf0).view(1, 1, -1).to(device)
                    dfs = [torch.FloatTensor(np.array(df)).view(1, 1, -1).to(device) for df in dfs]
                    if config.data.sine_f0_type == "cf0":
                        in_signal = signal_generator(cf0)
                    elif config.data.sine_f0_type == "f0":
                        in_signal = signal_generator(f0)
                    start = time()
                    c_y = model(c_x)
                    outs = vocoder(in_signal, c_y, dfs)
                    audio = outs[0].squeeze()
                    rtf = (time() - start) / (audio.size(-1) / config.data.sample_rate)
                    pbar.set_postfix({"RTF": rtf})
                    total_rtf += rtf

                    # save output signal as PCM 16 bit wav file
                    utt_id = feat_path.stem
                    fo = "_".join([f"{f:.2f}" for f in formants_factor])
                    spk_id = feat_path.parents[config.spkidx].name
                    save_dir = out_dir.joinpath(spk_id)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir.joinpath(f"{utt_id}_f{f0_factor:.2f}_fo{fo}.wav")
                    audio = audio.view(-1).cpu().numpy()
                    sf.write(save_path, audio, config.data.sample_rate, "PCM_16")

                # report average RTF
                mean_rtf = total_rtf / len(dataset)
                logger.info(f"Finished generation of {idx} utterances (RTF: {mean_rtf:.6f}, ×{1 / mean_rtf:.3f}).")


if __name__ == "__main__":
    main()
