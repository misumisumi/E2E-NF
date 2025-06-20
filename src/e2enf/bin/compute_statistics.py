import os
from logging import getLogger

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from joblib import dump, load
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from e2enf.features.f0_utils import logf0
from e2enf.utils.file_io import read_hdf5, read_txt

# A logger for this file
logger = getLogger(__name__)


def calc_stats(file_list, config):
    """Calcute statistics

    Args:
        file_list (list): File list.
        config (dict): Dictionary of config.

    """
    # define scalers
    scaler = load(config.stats) if os.path.isfile(config.stats) else {}
    for feat_type in config.feat_types:
        scaler[feat_type] = StandardScaler()

    # process over all of data
    for i, filename in tqdm(enumerate(file_list), total=len(file_list)):
        for feat_type in config.feat_types:
            if feat_type == "f0":
                f0 = read_hdf5(to_absolute_path(filename), "/f0")
                feat = np.expand_dims(f0[f0 > 0], axis=-1)
            elif "lcf" in feat_type or "lf0" in feat_type:  # log-f0 or lcf0, lcf1, lcf2
                continues_value = read_hdf5(to_absolute_path(filename), f"/{feat_type.replace('l', '')}")
                feat = logf0(continues_value)
            else:
                feat = read_hdf5(to_absolute_path(filename), f"/{feat_type}")
            if feat.shape[0] == 0:
                logger.warning(f"feat length is 0 {filename}/{feat_type}")
                continue
            scaler[feat_type].partial_fit(feat)

    if not os.path.exists(os.path.dirname(config.stats)):
        os.makedirs(os.path.dirname(config.stats))
    dump(scaler, to_absolute_path(config.stats))
    logger.info(f"Successfully saved statistics to {config.stats}.")


@hydra.main(version_base=None, config_path="config", config_name="compute_statistics")
def main(config: DictConfig):
    # show argument
    logger.info(OmegaConf.to_yaml(config))

    # read file list
    file_list = read_txt(to_absolute_path(config.feats))
    logger.info(f"number of utterances = {len(file_list)}")

    # calculate statistics
    calc_stats(file_list, config)


if __name__ == "__main__":
    main()
