from collections import defaultdict
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

# A logger for this file
logger = getLogger(__name__)


class BaseTrainer(object):
    """Customized trainer module for Source-Filter HiFiGAN training."""

    def __init__(
        self,
        config: dict,
        steps: int,
        epochs: int,
        data_loader: Dict[str, DataLoader],
        model: Dict[str, nn.Module],
        criterion: Dict[str, nn.Module],
        optimizer: Dict[str, torch.optim.Optimizer],
        scheduler: Dict[str, torch.optim.lr_scheduler._LRScheduler],
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
            early_stopping: EarlyStopping instance.
            device: Pytorch device instance.

        """
        self.config = config
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.check_intermediate_result = self.config.train.get("check_intermediate_result", False)

        self.scaler = torch.amp.GradScaler(enabled=config.train.amp.enabled)

        self.out_dir = Path(config.out_dir)
        self.finish_train = False
        self.writer = SummaryWriter(self.out_dir)
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

        if config.train.early_stopping.enabled:
            self.enaly_stopping = True
            self.patience = config.train.early_stopping.patience
            self.delta = config.train.early_stopping.delta
            self.target_loss = config.train.early_stopping.target_loss
            assert self.target_loss in self.criterion.keys(), (
                f"{self.target_loss} is not in criterion.\nTarget loss for early stopping must be in criterion"
            )
            self._counter: int = 0
            self._best_val_loss: Optional[float] = None
            self._early_stop: bool = False
        else:
            self.enaly_stopping = False

    def _train_step(self, batch):
        """Train model one step."""

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # eval code per step here

    @torch.no_grad()
    def _generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps, total=self.config.train.train_max_steps, dynamic_ncols=True)
        try:
            while True:
                # train one epoch
                self._train_epoch()

                # check whether training is finished
                if self.finish_train:
                    break
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Saving current state...")

        logger.info(f"finished: {self.finish_train}, {self.steps}, {self.config.train.train_max_steps}")
        # save last checkpoint
        self._check_log_interval(force=True)
        self._check_eval_interval(force=True)
        self._check_save_interval(force=True)

        self.tqdm.close()
        logger.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {"steps": self.steps, "epochs": self.epochs, "optimizer": {}, "scheduler": {}, "model": {}}
        for key, state in self.optimizer.items():
            state_dict["optimizer"][key] = state.state_dict()
        for key, state in self.scheduler.items():
            state_dict["scheduler"][key] = state.state_dict()
        for key, state in self.model.items():
            state_dict["model"][key] = state.state_dict()
        if self.config.train.amp.enabled:
            state_dict["scaler"] = self.scaler.state_dict()

        save_path = Path(checkpoint_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, checkpoint_path)

    def early_stopping(self):
        val_loss = self.total_eval_loss[self.target_loss]
        if torch.isnan(val_loss):
            logger.info("Validation loss is NaN. Ignoring this epoch.")
            return

        if self._best_val_loss is None:
            self._best_val_loss = val_loss
            self._check_save_interval(force=True)
        elif val_loss < self.__best_val_loss - self.delta:
            # Significant improvement detected
            self._best_val_loss = val_loss
            self._check_save_interval(force=True)
            self._counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self._counter += 1
            logger.info(f"EarlyStopping counter: {self._counter} out of {self._patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def load_checkpoint(self, checkpoint_path: PathLike, load_only_model: bool = False):
        """Load checkpoint.

        Args:
            checkpoint_path: Checkpoint path to be loaded.
            load_only_model: Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for key, model in self.model.items():
            if key not in state_dict["model"]:
                logger.warning(f"Key '{key}' not found in checkpoint. Skipping loading for this model.")
                continue
            model.load_state_dict(state_dict["model"][key])
        if not load_only_model:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            for key, optimizer in self.optimizer.items():
                if key not in state_dict["optimizer"]:
                    logger.warning(f"Key '{key}' not found in checkpoint. Skipping loading for this optimizer.")
                    continue
                optimizer.load_state_dict(state_dict["optimizer"][key])
            for key, scheduler in self.scheduler.items():
                if key not in state_dict["scheduler"]:
                    logger.warning(f"Key '{key}' not found in checkpoint. Skipping loading for this scheduler.")
                    continue
                scheduler.load_state_dict(state_dict["scheduler"][key])
            if self.config.train.amp.enabled and "scaler" in state_dict.keys():
                self.scaler.load_state_dict(state_dict["scaler"])

    def _train_epoch(self):
        """Train model one epoch."""
        self.tqdm.set_description(f"[train|epoch {self.epochs}]")
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # update counts
            self.steps += 1
            self.tqdm.update(1)
            self._check_train_finish()

            # check whether training is finished
            if self.finish_train:
                return

            # check intervals
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logger.info(f"(Steps: {self.steps}) Start evaluation.")

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["valid"], desc="[eval]"), 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                if "gen_and_save" in self.data_loader.keys():
                    loader = iter(self.data_loader["gen_and_save"])
                    batch = next(loader)
                self._generate_and_save_intermediate_result(batch)

        logger.info(f"(Steps: {self.steps}) Finished evaluation ({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logger.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self, force=False):
        if force or self.steps % self.config.train.save_interval_steps == 0:
            self.save_checkpoint(self.out_dir.joinpath("checkpoints", f"checkpoint-{self.steps}steps.pkl"))
            logger.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self, force=False):
        if force or self.steps % self.config.train.eval_interval_steps == 0:
            self._eval_epoch()
            if self.enaly_stopping:
                self._early_stopping()
            self.total_eval_loss = defaultdict(float)

    def _check_log_interval(self, force=False):
        if force or self.steps % self.config.train.log_interval_steps == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config.train.log_interval_steps
                logger.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config.train.train_max_steps:
            self.finish_train = True
