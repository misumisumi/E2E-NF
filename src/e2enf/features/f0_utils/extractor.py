from logging import getLogger

import numpy as np
import parselmouth
import pyreaper
import pyworld as pw
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .modules import MaskedAvgPool1d, MedianPool1d

logger = getLogger(__name__)


CREPE_RESAMPLE_KERNEL = {}
F0_KERNEL = {}


class F0_Extractor:
    def __init__(
        self,
        f0_extractor: str,
        sample_rate: int = 44100,
        hop_size: int = 512,
        f0_min: float = 65,
        f0_max: float = 800,
        fix_by_reaper: bool = False,
    ):
        self.f0_extractor = f0_extractor
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        if f0_extractor == "crepe":
            import torchcrepe

            key_str = str(sample_rate)
            if key_str not in CREPE_RESAMPLE_KERNEL:
                CREPE_RESAMPLE_KERNEL[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
        if f0_extractor == "rmvpe":
            if "rmvpe" not in F0_KERNEL:
                from audiometrics.encoder.rmvpe import RMVPE

                F0_KERNEL["rmvpe"] = RMVPE("pretrain/rmvpe/model.pt", hop_length=160)
            self.rmvpe = F0_KERNEL["rmvpe"]
        if f0_extractor == "fcpe":
            import torchfcpe

            self.device_fcpe = "cuda" if torch.cuda.is_available() else "cpu"
            if "fcpe" not in F0_KERNEL:
                from torchfcpe import spawn_bundled_infer_model

                F0_KERNEL["fcpe"] = spawn_bundled_infer_model(device=self.device_fcpe)
            self.fcpe = F0_KERNEL["fcpe"]
        if f0_extractor in ["parselmouth", "dio", "harvest"]:
            self.fix_by_reaper = fix_by_reaper
            if self.fix_by_reaper:
                self.hop_size /= 2
                logger.info("extracting f0 will be fixed by reaper.")
        else:
            self.fix_by_reaper = False
            logger.warning("can be used with reaper only when using parselmouth, dio, harvest.")

    def extract(
        self,
        audio: np.ndarray,
        uv_interp: bool = False,
        device: str = None,
        silence_front: int = 0,
        return_time: bool = False,
    ) -> np.ndarray:  # audio: 1d numpy array
        # extractor start time
        n_frames = int(len(audio) // self.hop_size) + 1

        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)) :]

        # extract f0 using parselmouth
        if self.f0_extractor == "parselmouth":
            l_pad = int(np.ceil(1.5 / self.f0_min * self.sample_rate))
            r_pad = int(self.hop_size * ((len(audio) - 1) // self.hop_size + 1) - len(audio) + l_pad + 1)
            s = parselmouth.Sound(np.pad(audio, (l_pad, r_pad)), self.sample_rate).to_pitch_ac(
                time_step=self.hop_size / self.sample_rate,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
            )
            assert np.abs(s.t1 - 1.5 / self.f0_min) < 0.001
            f0 = np.pad(s.selected_array["frequency"], (start_frame, 0))
            if len(f0) < n_frames:
                f0 = np.pad(f0, (0, n_frames - len(f0)))
            f0 = f0[:n_frames]

        # extract f0 using dio
        elif self.f0_extractor == "dio":
            _f0, t = pw.dio(
                audio.astype("double"),
                self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                channels_in_octave=2,
                frame_period=(1000 * self.hop_size / self.sample_rate),
            )
            f0 = pw.stonemask(audio.astype("double"), _f0, t, self.sample_rate)
            f0 = np.pad(f0.astype("float"), (start_frame, n_frames - len(f0) - start_frame))

        # extract f0 using harvest
        elif self.f0_extractor == "harvest":
            f0, t = pw.harvest(
                audio.astype("double"),
                self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                frame_period=(1000 * self.hop_size / self.sample_rate),
            )
            f0 = np.pad(f0.astype("float"), (start_frame, n_frames - len(f0) - start_frame))

        elif self.f0_extractor == "reaper":
            # convert audio to int16 if float32
            if np.issubdtype(audio.dtype, np.floating):
                audio = (audio * 32768).astype(np.int16)

            _, _, time, f0, _ = pyreaper.reaper(
                audio,
                self.sample_rate,
                frame_period=self.hop_size / self.sample_rate,
                minf0=self.f0_min,
                maxf0=self.f0_max,
            )
            f0 = np.pad(f0.astype("float"), (start_frame, n_frames - len(f0) - start_frame))
            f0 = np.where(f0 == -1.0, 0, f0)

        # extract f0 using crepe
        elif self.f0_extractor == "crepe":
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            resample_kernel = self.resample_kernel.to(device)
            wav16k_torch = resample_kernel(torch.FloatTensor(audio).unsqueeze(0).to(device))

            f0, pd = torchcrepe.predict(
                wav16k_torch,
                16000,
                80,
                self.f0_min,
                self.f0_max,
                pad=True,
                model="full",
                batch_size=512,
                device=device,
                return_periodicity=True,
            )
            pd = MedianPool1d(pd, 4)
            f0 = torchcrepe.threshold.At(0.05)(f0, pd)
            f0 = MaskedAvgPool1d(f0, 4)

            f0 = f0.squeeze(0).cpu().numpy()
            f0 = np.array(
                [
                    f0[int(min(int(np.round(n * self.hop_size / self.sample_rate / 0.005)), len(f0) - 1))]
                    for n in range(n_frames - start_frame)
                ]
            )
            f0 = np.pad(f0, (start_frame, 0))

        # extract f0 using rmvpe
        elif self.f0_extractor == "rmvpe":
            f0 = self.rmvpe.infer_from_audio(audio, self.sample_rate, device=device, threshold=0.03, use_viterbi=False)
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))

        # extract f0 using fcpe
        elif self.f0_extractor == "fcpe":
            _audio = torch.from_numpy(audio).to(self.device_fcpe).unsqueeze(0)
            f0 = self.fcpe(_audio, sr=self.sample_rate, decoder_mode="local_argmax", threshold=0.006)
            f0 = f0.squeeze().cpu().numpy()
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))

        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")

        if self.fix_by_reaper:
            if np.issubdtype(audio.dtype, np.floating):
                audio = (audio * 32768).astype(np.int16)
            _, _, _time, f0_mask, _ = pyreaper.reaper(
                audio,
                self.sample_rate,
                frame_period=self.hop_size / self.sample_rate,
                minf0=self.f0_min,
                maxf0=self.f0_max,
            )
            f0 = f0[1::2].copy(order="C")
            t = t[1::2].copy(order="C")
            f0_mask = f0_mask[1::2]
            f0_mask = np.pad(f0_mask.astype("float"), (start_frame, n_frames - len(f0) - start_frame))
            f0 = np.where(f0_mask == -1.0, 0, f0).copy(order="C")
            t[: _time.shape[0]] = _time

        # interpolate the unvoiced f0
        if uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min

        if return_time:
            if self.f0_extractor in ["dio", "harvest"]:
                return f0, t
            else:
                return f0, np.arange(n_frames) * self.hop_size / self.sample_rate
        else:
            return f0
