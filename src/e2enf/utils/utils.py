import copy
import os
from logging import getLogger
from pathlib import Path
from typing import Optional

logger = getLogger(__name__)


def spk_division(
    file_list: list[os.PathLike], config: dict, spkinfo: dict, target: Optional[str] = None
) -> list[tuple[os.PathLike, dict]]:
    """Divide list into speaker-dependent list

    Args:
        file_list: Waveform list
        config: Config
        spkinfo: Dictionary of
            speaker-dependent f0 range and power threshold
        split: Path split string
        target: Target key in config

    Return:

    """
    file_and_conf_list = []
    per_spk_conf = {}
    for file in file_list:
        spk = Path(file).parents[config.spkidx].name
        if per_spk_conf.get(spk) is not None:
            tempc = per_spk_conf[spk]
        else:
            tempc = copy.deepcopy(config)
            if spk in spkinfo:
                for key in spkinfo[spk].keys():
                    tempc[key] = spkinfo[spk][key]
                per_spk_conf[spk] = tempc
            else:
                msg = f"Since {spk} is not in spkinfo dict, "
                msg += "use default settings."
                logger.info(msg)
                per_spk_conf[spk] = tempc
        file_and_conf_list.append((file, tempc))

    return file_and_conf_list
