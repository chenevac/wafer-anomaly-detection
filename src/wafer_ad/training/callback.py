import logging
import operator
import os
from typing import Callable, Dict, Optional

import numpy as np


class Callback:
    """Abstract base class used to build new callbacks."""

    def on_fit_start(self, trainer) -> None:
        """Called when fit begins."""

    def on_fit_end(self, trainer) -> None:
        """Called when fit ends."""

    def on_train_epoch_start(self, trainer) -> None:
        """Called when the train epoch begins."""

    def on_train_epoch_end(self, trainer) -> None:
        """Called when the train epoch ends."""

    def on_validation_epoch_start(self, trainer) -> None:
        """Called when the train epoch begins."""

    def on_validation_epoch_end(self, trainer) -> None:
        """Called when the train epoch ends."""
    
    
class EarlyStopping(Callback):
    """The EarlyStopping callback can be used to monitor a validation metric
    and stop the training when no improvement is observed."""

    mode_dict = {"min": operator.lt, "max": operator.gt}

    def __init__(
        self,
        monitor: str,
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        """
        Example:
            >>> from onlinelearning.trainer import Trainer
            >>> from onlinelearning.callbacks import EarlyStopping
            >>> early_stopping = EarlyStopping('val_loss')
            >>> trainer = Trainer(callbacks=[early_stopping])

        Args:
            monitor: quantity to be monitored.
            patience: number of checks with no improvement after which
                training will be stopped.
            min_delta: minimum change in the monitored quantity to qualify
                as an improvement, i.e. an absolute change of less than or
                equal to `min_delta`, will count as no improvement.
            mode: one of ``'min'``, ``'max'``. In ``'min'`` mode, training
                will stop when the quantity monitored has stopped decreasing
                and in ``'max'`` mode it will stop when the quantity monitored
                has stopped increasing.
        """
        super().__init__()

        if patience < 1:
            raise ValueError("Argument patience should be a stricly positive integer.")
        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")
        if mode not in self.mode_dict:
            raise ValueError(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {mode}")

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.wait_count = 0
        self.min_delta *= 1 if self.mode == "max" else -1
        self.best_score = np.Inf if self.mode == "min" else -np.Inf

    def reset_wait_count_and_best_score(self) -> None:
        self.wait_count = 0
        self.best_score = np.Inf if self.mode == "min" else -np.Inf

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def _validate_condition_metric(self, metrics: Dict[str, float]) -> None:
        monitor_val = metrics.get(self.monitor)
        if monitor_val is None:
            error_msg = "Early stopping conditioned on metric "\
                        f"`{self.monitor}` which is not available. Pass in "\
                        "or modify your `EarlyStopping` callback to use any "\
                        f"of the following: "\
                        "`{'`, `'.join(list(metrics.keys()))}`"
            raise RuntimeError(error_msg)

    def on_validation_epoch_end(self, trainer) -> None:
        self._validate_condition_metric(trainer.metrics_history)

        score = trainer.metrics_history[self.monitor][-1]
        if self.monitor_op(score - self.min_delta, self.best_score):
            logging.info(
                "Monitored metric %s improved (%s --> %s).",
                self.monitor,
                self.best_score,
                score,
            )
            self.best_score = score
            self.wait_count = 0
        else:
            self.wait_count += 1
            logging.info(
                "Monitored metric %s not improved (%s --> %s).",
                self.monitor,
                self.best_score,
                score,
            )
            logging.info(
                "EarlyStopping counter: %s out of %s",
                self.wait_count,
                self.patience,
            )
            if self.wait_count >= self.patience:
                trainer.should_stop = True
                logging.info(
                    "Monitored metric %s did not improve in the last %s"
                    " records.\nBest score %s. Signaling Trainer to stop.",
                    self.monitor,
                    self.wait_count,
                    self.best_score,
                )
        
        
class ModelCheckpoint(Callback):
    """
    Save the model periodically by monitoring a quantity.
    """

    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = "pth"  ###
    STARTING_VERSION = 0

    mode_dict = {"min": operator.lt, "max": operator.gt}

    def __init__(
        self,
        dirpath: str,
        filename: str,
        monitor: Optional[str] = None,
        save_last: bool = False,
        mode: str = "min",
        every_n_epochs: Optional[int] = None,
    ):
        """
        Args:
            dirpath: directory to save the model file.
            filename: checkpoint filename.
            monitor: quantity to monitor. By default it is ``None`` which
                saves a checkpoint only for the last epoch.
            every_n_epochs: Number of epochs between checkpoints. This value
                must be ``None`` or non-negative.
        """
        super().__init__()
        if every_n_epochs is not None and every_n_epochs < 1:
            raise ValueError("Argument every_n_epochs should be None or a stricly positive integer.")
        if mode not in self.mode_dict:
            raise ValueError(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {mode}")
        self.monitor = monitor
        self.save_last = save_last
        self._every_n_epochs: Optional[int] = every_n_epochs
        self.dirpath = dirpath
        self.filename = filename
        self.current_version = self.STARTING_VERSION + 1

        self.mode = mode
        self.best_score = np.Inf if self.mode == "min" else -np.Inf

    def reset_best_score(self) -> None:
        self.best_score = np.Inf if self.mode == "min" else -np.Inf

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        trainer.save_checkpoint(filepath)

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def on_validation_epoch_end(self, trainer) -> None:
        """Save a checkpoint at the end of the validation stage."""
        should_save_checkpoint_every_n_epochs = False
        if self._every_n_epochs is not None and trainer.current_epoch % self._every_n_epochs == 0:
            should_save_checkpoint_every_n_epochs = True

        self._validate_condition_metric(trainer.metrics_history)

        score = trainer.metrics_history[self.monitor][-1]
        should_save_checkpoint_monitor = False
        if self.monitor_op(score, self.best_score):
            should_save_checkpoint_monitor = True
            self.best_score = score

        should_save_checkpoint = (
            should_save_checkpoint_every_n_epochs or
            should_save_checkpoint_monitor
        )
        if should_save_checkpoint:
            filepath = self.format_checkpoint_name(ver=self.current_version)
            self._save_checkpoint(trainer, filepath)

    def on_fit_end(self, trainer) -> None:
        if self.save_last:
            filepath = self.format_checkpoint_name(name_extension=self.CHECKPOINT_NAME_LAST)
            self._save_checkpoint(trainer, filepath)
        self.current_version += 1

    def format_checkpoint_name(self, name_extension: Optional[str] = None, ver: Optional[int] = None) -> str:
        """Generate a filename according to the defined template."""
        filename = self.filename
        if "." in filename:  # todo : this may cause errors
            logging.warning("Extension detected in filename. Then removed.")
            filename = filename.split(".")[0]
        if name_extension is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join([filename, name_extension])
        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))
        filename += f".{self.FILE_EXTENSION}"
        return os.path.join(self.dirpath, filename)

    def _validate_condition_metric(self, metrics: Dict[str, float]) -> None:
        monitor_val = metrics.get(self.monitor)
        if monitor_val is None:
            error_msg = f"Model checkpoint on metric `{self.monitor}` which"\
                " is not available. Pass in or modify your `ModelCheckpoint`"\
                " callback to use any of the following: "\
                f"`{'`, `'.join(list(metrics.keys()))}`"
            raise RuntimeError(error_msg)
