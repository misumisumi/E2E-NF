import os
import sys
from logging import getLogger

import hydra
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from e2enf.datasets import FeatDataset, SingleFMCollator
from e2enf.utils.trainer import BaseTrainer

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


# A logger for this file
logger = getLogger(__name__)


class Trainer(BaseTrainer):
    """Customized trainer module for NeuralFormants training."""

    def __init__(
        self,
        config: dict,
        steps: int,
        epochs: int,
        data_loader: dict,
        model: dict,
        criterion: dict,
        optimizer: dict,
        scheduler: dict,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            config: Config dict loaded from yaml format configuration file.
            steps: Initial global steps.
            epochs: Initial global epochs.
            data_loader: Dict of data loaders. It must constrain "train" and "dev" loaders.
            model: Dict of models. It must constrain "generator" and "discriminator" models.
            criterion: Dict of criterions. It must constrain "adv", "encode" and "f0" criterions.
            optimizer: Dict of optimizers. It must constrain "generator" and "discriminator" optimizers.
            scheduler: Dict of schedulers. It must constrain "generator" and "discriminator" schedulers.
            device: Pytorch device instance.

        """
        super().__init__(
            config=config,
            steps=steps,
            epochs=epochs,
            data_loader=data_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        _, c, mfbsp = batch
        if self.config.train.mfbsp2spkparams:
            x = mfbsp.to(self.device)
            y = c.to(self.device)
        else:
            x = c.to(self.device)
            y = mfbsp.to(self.device)

        # generator forward
        with torch.autocast(device_type=self.device.type, enabled=self.config.train.amp.enabled):
            y_hat = self.model["feature-mapping"](x)

            # calculate spectral loss
            loss = self.criterion["loss"](y, y_hat)
            self.total_train_loss["train/mse_loss"] += loss.item()

        # update generator
        self.optimizer["feature-mapping"].zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer["feature-mapping"])
        if self.config.train.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["feature-mapping"].parameters(),
                self.config.train.grad_norm,
            )
        self.scaler.step(self.optimizer["feature-mapping"])
        self.scaler.update()
        self.scheduler["feature-mapping"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        _, c, mfbsp = batch
        if self.config.train.mfbsp2spkparams:
            x = mfbsp.to(self.device)
            y = c.to(self.device)
        else:
            x = c.to(self.device)
            y = mfbsp.to(self.device)

        # generator forward
        y_hat = self.model["feature-mapping"](x)

        # calculate spectral loss
        loss = self.criterion["loss"](y, y_hat)
        self.total_train_loss["eval/mse_loss"] += loss.item()

    @torch.no_grad()
    def _generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        if not self.config.train.mfbsp2spkparams:
            # parse batch
            _, c, mfbsp = batch
            if self.config.train.mfbsp2spkparams:
                x = mfbsp.to(self.device)
            else:
                x = c.to(self.device)

            # generator forward
            y_hat = self.model["feature-mapping"](x)
            for i, _y_hat in enumerate(y_hat):
                _y_hat = _y_hat.squeeze(0)
                _y_hat = librosa.amplitude_to_db(np.exp(_y_hat.cpu().numpy()), ref=np.max)
                # _y_hat = _y_hat.cpu().numpy()
                fig = plt.figure(figsize=(8, 6))
                librosa.display.specshow(
                    _y_hat,
                    y_axis="mel",
                    x_axis="time",
                    sr=self.config.data.sample_rate,
                    hop_length=self.config.data.hop_size,
                    cmap="viridis",
                )
                # plt.ylim([spectrogram.ymin, spectrogram.ymax])
                plt.xlabel("time [s]")
                plt.ylabel("frequency [Hz]")
                self.writer.add_figure(f"spectrogram/gen_{i}", fig, self.steps)
                plt.clf()
                plt.close()


@hydra.main(version_base=None, config_path="config", config_name="train_fm")
def main(config: DictConfig) -> None:
    """Run training process."""

    if config.device is not None:
        print(f"Device: {config.device}")
        device = torch.device(config.device)
    elif not torch.cuda.is_available():
        print("CPU")
        device = torch.device("cpu")
    else:
        print("GPU")
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

    # fix seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # check directory existence
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # write config to yaml file
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))
    logger.info(OmegaConf.to_yaml(config))

    train_dataset = FeatDataset(
        feat_lists=config.data.train_feat,
        stats_lists=config.data.stats,
        feat_length_threshold=config.data.batch_max_frames,
        allow_cache=config.data.allow_cache,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        aux_feats=config.data.aux_feats,
    )
    logger.info(f"The number of training files = {len(train_dataset)}.")

    valid_dataset = FeatDataset(
        feat_lists=config.data.valid_feat,
        stats_lists=config.data.stats,
        feat_length_threshold=config.data.batch_max_frames,
        allow_cache=config.data.allow_cache,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        aux_feats=config.data.aux_feats,
    )
    logger.info(f"The number of validation files = {len(valid_dataset)}.")

    dataset = {"train": train_dataset, "valid": valid_dataset}

    collator = SingleFMCollator(batch_max_frames=config.data.batch_max_frames)

    train_sampler, valid_sampler = None, None
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=True,
            collate_fn=collator,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=train_sampler,
            pin_memory=config.data.pin_memory,
        ),
        "valid": DataLoader(
            dataset=dataset["valid"],
            shuffle=True,
            collate_fn=collator,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=valid_sampler,
            pin_memory=config.data.pin_memory,
        ),
    }

    # define models and optimizers
    model = {
        "feature-mapping": hydra.utils.instantiate(config.model).to(device),
    }

    # define training criteria
    criterion = {
        "loss": hydra.utils.instantiate(config.train.loss).to(device),
    }

    # # define optimizers and schedulers
    optimizer = {
        "feature-mapping": hydra.utils.instantiate(
            config.train.neuralformants_optimizer, params=model["feature-mapping"].parameters()
        ),
    }
    scheduler = {
        "feature-mapping": hydra.utils.instantiate(
            config.train.neuralformants_scheduler, optimizer=optimizer["feature-mapping"]
        ),
    }

    # define trainer
    trainer = Trainer(
        config=config,
        steps=0,
        epochs=0,
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # load trained parameters from checkpoint
    if config.train.resume:
        resume = os.path.join(config.out_dir, "checkpoints", f"checkpoint-{config.train.resume}steps.pkl")
        if os.path.exists(resume):
            trainer.load_checkpoint(resume)
            logger.info(f"Successfully resumed from {resume}.")
        else:
            logger.info(f"Failed to resume from {resume}.")
            sys.exit(0)
    else:
        logger.info("Start a new training process.")

    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(os.path.join(config.out_dir, "checkpoints", f"checkpoint-{trainer.steps}steps.pkl"))
        logger.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
