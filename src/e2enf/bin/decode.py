import os
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


@hydra.main(version_base=None, config_path="config", config_name="decode")
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
    out_dir = Path(config.out_dir)
    if config.checkpoint_path is None:
        checkpoint_path = out_dir.joinpath(
            "checkpoints",
            f"checkpoint-{config.checkpoint_steps}steps.pkl",
        )
    else:
        checkpoint_path = config.checkpoint_path
    state_dict = torch.load(to_absolute_path(checkpoint_path), map_location="cpu", weights_only=False)
    logger.info(f"Loaded model parameters from {checkpoint_path}.")
    model = hydra.utils.instantiate(config.generator)
    model.load_state_dict(state_dict["model"]["generator"])
    model.remove_weight_norm()
    model.eval().to(device)

    # check directory existence
    out_dir = Path(to_absolute_path(out_dir.joinpath("wav", str(config.checkpoint_steps))))
    out_dir.mkdir(parents=True, exist_ok=True)

    for f0_factor in config.f0_factors:
        for formants_factor in config.formants_factors:
            dataset = FeatDataset(
                feat_lists=config.data.eval_feat,
                stats_lists=config.data.stats,
                return_filename=True,
                sample_rate=config.data.sample_rate,
                hop_size=config.data.hop_size,
                aux_feats=config.data.aux_feats,
                f0_factor=f0_factor,
                formants_factor=formants_factor,
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
                    _, feat_path, c, _, f0, cf0 = items
                    # create dense factors
                    dfs = []
                    for df, us in zip(
                        config.data.dense_factors,
                        np.cumprod(config.generator.upsample_scales),
                    ):
                        dfs += [
                            np.repeat(dilated_factor(cf0, config.data.sample_rate, df), us)
                            if config.data.df_f0_type == "cf0"
                            else np.repeat(dilated_factor(f0, config.data.sample_rate, df), us)
                        ]
                    c = torch.FloatTensor(c).unsqueeze(0).transpose(2, 1).to(device)
                    f0 = torch.FloatTensor(f0).view(1, 1, -1).to(device)
                    cf0 = torch.FloatTensor(cf0).view(1, 1, -1).to(device)
                    dfs = [torch.FloatTensor(np.array(df)).view(1, 1, -1).to(device) for df in dfs]
                    if config.data.sine_f0_type == "cf0":
                        in_signal = signal_generator(cf0)
                    elif config.data.sine_f0_type == "f0":
                        in_signal = signal_generator(f0)

                    # perform decoding
                    start = time()
                    outs = model(in_signal, c, dfs)
                    y = outs[0]
                    rtf = (time() - start) / (y.size(-1) / config.data.sample_rate)
                    pbar.set_postfix({"RTF": rtf})
                    total_rtf += rtf

                    # save output signal as PCM 16 bit wav file
                    utt_id = feat_path.stem
                    fo = "_".join([f"{f:.2f}" for f in formants_factor])
                    spk_id = feat_path.parents[config.spkidx].name
                    save_dir = out_dir.joinpath(spk_id)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir.joinpath(f"{utt_id}_f{f0_factor:.2f}_fo{fo}.wav")
                    y = y.view(-1).cpu().numpy()
                    sf.write(save_path, y, config.data.sample_rate, "PCM_16")

                    # save source signal as PCM 16 bit wav file
                    if config.save_source:
                        save_path = save_path.replace(".wav", "_s.wav")
                        s = outs[1].view(-1).cpu().numpy()
                        s = s / np.max(np.abs(s))  # normalize
                        sf.write(save_path, s, config.data.sample_rate, "PCM_16")

                # report average RTF
                mean_rtf = total_rtf / len(dataset)
                logger.info(f"Finished generation of {idx} utterances (RTF: {mean_rtf:.6f}, ×{1 / mean_rtf:.3f}).")


if __name__ == "__main__":
    main()
