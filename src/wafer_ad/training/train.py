

import logging
import os
from pathlib import Path
import time
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
import torch
from wafer_ad.data.dataloader import get_data_loaders
from wafer_ad.models.flow import MSFlowModel
from wafer_ad.training.callback import Callback, EarlyStopping, ModelCheckpoint
from wafer_ad.training.loss import MSFlowLoss
from wafer_ad.training.metric import AverageMeter
from wafer_ad.utils.config import Config
from wafer_ad.utils.constant import PROJECT_ROOT
from wafer_ad.utils.devices import get_device
from wafer_ad.utils.seed import init_seeds


class Trainer:
    def __init__(
        self,
        device: Optional[str] = None,
        enable_progress_bar: bool = True,
        learning_rate: float = 1e-4,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_norm_type: float = 2.,
        dirpath_model_checkpoint: str = "/kaggle/working/"
    ) -> None:
        """_summary_

        Args:
            device: The device to use. If None, will use GPU if available.
            gradient_clip_val: The value at which to clip gradients.
                Passing ``gradient_clip_val=None`` disables.
            accumulate_grad_batches: accumulate_grad_batches: Accumulates grads every k batches.
            learning_rate: Learning rate for the optimizer.
            enable_progress_bar: Whether to enable the progress bar.
        """
        logging.info("Initializing trainer.")
        self.gradient_clip_val: Optional[Union[int, float]] = gradient_clip_val
        self.gradient_clip_norm_type = gradient_clip_norm_type
        self.enable_progress_bar: bool = enable_progress_bar
        self.device: str = device if device is not None else get_device()
        self.accumulate_grad_batches: int = accumulate_grad_batches
        
        # TODO: make optimizer, schedulers and callbacks configurable
        self.optimizer = torch.optim.Adam
        self.optimizer_kwargs: Dict[str, Any] = {"lr": learning_rate}
        
        self.schedulers: List[torch.optim.lr_scheduler._LRScheduler] = [
            torch.optim.lr_scheduler.LinearLR,
            torch.optim.lr_scheduler.MultiStepLR
        ]
        self.schedulers_kwargs: List[Dict[str, Any]] = [
            # Warmup sur les 500 premiers steps
            {
                "start_factor": 0.1,
                "end_factor": 1.0,
                "total_iters": 500,
            },
            # Décroissance après certains steps
            {
                "milestones": [5_000, 10_000, 20_000],
                "gamma": 0.1,
            },
        ]
        
        self.callbacks: List[Callback] = [
            ModelCheckpoint,
            EarlyStopping,
        ]
        self.callbacks_kwargs: List[Dict[str, Any]] = [
            {
                "dirpath": dirpath_model_checkpoint,
                "filename": "csflow",
                "every_n_epochs": 4,
            },
            {
                "monitor": "val_loss"
            }
        ]
        
        self.loss_fn = MSFlowLoss()
        
        self.metrics = []
        self.metrics_kwargs = []
        self.metrics_history: Dict[str, List[float]]
        
        self.start, self.end = None, None
        self.should_stop: bool = False
        self.current_epoch: int
        self.model: Optional[torch.nn.Module] = None
        self.trainable_params: Optional[List[torch.nn.Parameter]] = None

        logging.info("Trainer successfully initialized.")
        
    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Fit one epoch.

        Iterates over all batches in the dataloader (one epoch).
        """
        self._call_callbacks("on_train_epoch_start")
        self._reset_metrics()
        logging.info(
            "Epoch %s\n-------------------------------",
            self.current_epoch
        )
        if self.enable_progress_bar:
            dataloader = tqdm(dataloader, total=len(dataloader))
        
        self.model.train()
        total_loss = AverageMeter()
        for batch_idx, (images, _, _) in enumerate(dataloader):
            images = images.to(self.device)
            z_list, jac = self.model(images)  # Forward pass
            loss = self.loss_fn(z_list, jac)
            total_loss.update(loss.item(), n=images.size(0))
            if self.enable_progress_bar:
                dataloader.set_description(f'loss: {total_loss.avg:.2f}')
                
            loss = loss / self.accumulate_grad_batches  # Normalize the gradients
            loss.backward()
            if ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (batch_idx + 1 == len(dataloader)):
                # gradient clipping
                if self.gradient_clip_val is not None:  # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_params,
                        max_norm=self.gradient_clip_val,
                        norm_type=self.gradient_clip_norm_type,
                    )
                # Update Optimizer
                self.optimizer.step()
                self.optimizer.zero_grad()  # Reset gradients
                
                for scheduler in self.schedulers:  # Update LR Schedulers
                    scheduler.step()
        self._update_train_metrics_history(total_loss.avg)    
        self._call_callbacks("on_train_epoch_end")
        
    def eval(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Evaluate the dataloader."""
        self._call_callbacks("on_validation_epoch_start")
        self.model.eval()
        total_loss = AverageMeter()
        self._reset_metrics()

        if self.enable_progress_bar:
            dataloader = tqdm(dataloader, total=len(dataloader))
        for batch_idx, (images, _, _) in enumerate(dataloader):
            images = images.to(self.device)
            with torch.no_grad():
                z_list, jac = self.model(images)  # Forward pass
                loss = self.loss_fn(z_list, jac)
                total_loss.update(loss.item(), n=images.size(0))
                if self.enable_progress_bar:
                    dataloader.set_description(f'loss: {total_loss.avg:.2f}')
        self._update_val_metrics_history(total_loss.avg)
        self._call_callbacks("on_validation_epoch_end")


    def _reset_metrics(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics:
            metric.reset()

    def _update_metrics(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        """Update all metrics with new predictions and targets."""
        for metric in self.metrics:
            metric.update(y_hat, y)

    def _update_train_metrics_history(self, loss: float) -> None:
        """Update the training metrics history."""
        for metric in self.metrics:
            self.metrics_history[f"train_{metric.__class__.__name__}"].append(float(metric.compute()))
        self.metrics_history["train_loss"].append(loss)

    def _update_val_metrics_history(self, loss: float) -> None:
        """Update the validation metrics history."""
        for metric in self.metrics:
            self.metrics_history[f"val_{metric.__class__.__name__}"].append(float(metric.compute()))
        self.metrics_history["val_loss"].append(loss)

    def _reset_metrics_history(self) -> None:
        """Reset the metrics history dictionary."""
        self.metrics_history = {"train_loss": [], "val_loss": []}
        for m in self.metrics:
            self.metrics_history.update(
                {
                    f"train_{m.__class__.__name__}": [],
                    f"val_{m.__class__.__name__}": [],
                }
            )

    def train(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: Optional[torch.utils.data.DataLoader],
        n_epochs: int = 100,
        evaluate_every_n_epochs: int = 1,
    ) -> None:
        """
        Runs the full optimization routine.

        Args:
            model: Model to fit.
            train_dataloader: A :class:`torch.utils.data.DataLoader`
                specifying training samples.
            val_dataloader: A :class:`torch.utils.data.DataLoader`
                specifying validation samples.
            evaluate_every_n_epochs: Evaluate every n epochs on validation set.
            n_epochs: Number of epochs to fit.
        """
        if valid_dataloader is None:
            logging.warning("No validation dataloader specified for training. ☠️")
        self.on_fit_start(model)
        for _ in range(n_epochs):
            self.train_one_epoch(train_dataloader)
            if valid_dataloader is not None and (self.current_epoch % evaluate_every_n_epochs == 0):
                self.eval(valid_dataloader)
            info_metric = " | ".join(
                [f"{k}: {v[-1]:.4f}" for k, v in self.metrics_history.items()]
            )
            logging.info("Epoch %s  %s\n", self.current_epoch, info_metric)
            self.current_epoch += 1
            if self.should_stop:
                break
        self.on_fit_end()
       
    def _resolve_uninstanciated(self, to_be_resolved: str) -> None:

        uninstanciated_objects_list: List[Union[Any, type]] = [c for c in getattr(self, to_be_resolved) if isinstance(c, type)]
        logging.info("%s uninstanciated %s detected.",len(uninstanciated_objects_list), to_be_resolved)  
        assert len(uninstanciated_objects_list) == len(getattr(self, to_be_resolved + "_kwargs")), f"Number of uninstanciated {to_be_resolved} and number of callbacks kwargs not equals"
        c_kwargs: int = 0
        for c_class in range(len(getattr(self, to_be_resolved))):
            if isinstance(getattr(self, to_be_resolved)[c_class], type):
                logging.info("Instanciation of %s %s", to_be_resolved[:-1], getattr(self, to_be_resolved)[c_class])
                if to_be_resolved == "schedulers":
                    getattr(self, to_be_resolved)[c_class] = getattr(self, to_be_resolved)[c_class](self.optimizer, **getattr(self, to_be_resolved + "_kwargs")[c_kwargs])
                else:
                    getattr(self, to_be_resolved)[c_class] = getattr(self, to_be_resolved)[c_class](**getattr(self, to_be_resolved + "_kwargs")[c_kwargs])
                c_kwargs += 1
        setattr(self, to_be_resolved + "_kwargs", list())
        
    def on_fit_start(self, model: torch.nn.Module) -> None:
        logging.info("Start fitting the model.")
        self.start = time.time()
        
        self.current_epoch = 1
        self.should_stop = False
        
        self.model = model
        self.model.to(self.device)
        
        self.trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        
        self._reset_metrics_history()

        if isinstance(self.optimizer, type):
            logging.info("Instanciation of optimizer %s", self.optimizer)
            self.optimizer = self.optimizer(self.trainable_params, **self.optimizer_kwargs)
        self.optimizer.zero_grad()
        
        self._resolve_uninstanciated("callbacks")
        self._resolve_uninstanciated("schedulers")
        self._resolve_uninstanciated("metrics")

        self._call_callbacks("on_fit_start")

    def on_fit_end(self) -> None:
        self._call_callbacks("on_fit_end")
        self.end = time.time()
        time_delta = round((self.end - self.start) / 60, 2)
        logging.info("Model successfully trained | %s epochs | %s min", self.current_epoch - 1, time_delta,)
        
    def _call_callbacks(self, call_name: str) -> None:
        for callback in self.callbacks:
            fn = getattr(callback, call_name)
            fn(self)
        
    def save_checkpoint(self, filepath: str) -> None:
        if self.model is None:
            raise AttributeError(
                "Saving a checkpoint is only possible if a model is attached "
                "to the Trainer. Did you call `Trainer.save_checkpoint()` "
                "before calling `Trainer.{fit,test}`?"
            )
        self.model.save_state_dict(filepath)
       
    @classmethod 
    def from_config(cls, source: Union[str, Config]) -> "Trainer":
        """Construct `Model` instance from `source` configuration."""
        logging.info("Creating Trainer from config.")
        if isinstance(source, str):
            source = Config.from_yaml(source)
        return cls(
            device=source.device,
            enable_progress_bar=source.enable_progress_bar,
            learning_rate=source.learning_rate,
            accumulate_grad_batches=source.accumulate_grad_batches,
            gradient_clip_val=source.gradient_clip_val,
            gradient_clip_norm_type=source.gradient_clip_norm_type,
        )
    
      
      
if __name__ == "__main__":
    init_seeds(42)
    project_root = Path(__file__).resolve().parents[3]
    model = MSFlowModel.from_config(os.path.join(project_root, "configs", "model.yaml"))
    train_loader, val_loader, test_loader = get_data_loaders(idx_dataset=1)
    trainer = Trainer()
    
    trainer.train(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
    )

