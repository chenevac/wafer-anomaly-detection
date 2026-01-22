

import logging
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
import torch
from torch.optim import Optimizer
from wafer_ad.training.callback import Callback
from wafer_ad.training.metric import AverageMeter
from wafer_ad.utils.configuration.configurable import Configurable
from wafer_ad.utils.configuration.training_config import TrainingConfig
from wafer_ad.utils.device import get_device
from wafer_ad.utils.other import nullable_union_list_object_to_list
from wafer_ad.utils.path import resolve_path
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler

class Trainer(Configurable):
    def __init__(
        self,
        optimizer: Union[Optimizer, type],
        loss_fn: Union[_Loss, type],
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
        schedulers: Optional[Union[LRScheduler, List[LRScheduler]]] = None,
        
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn_kwargs: Optional[Dict[str, Any]] = None,
        callbacks_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        schedulers_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_norm_type: float = 2.,

        accumulate_grad_batches: int = 1,
        device: Optional[str] = None,
        enable_progress_bar: bool = True,
    ) -> None:
        """Initializes the Trainer.

        Args:
            optimizer: The optimizer to use.
            loss_fn: The loss function to use.
            callbacks: A list of callbacks to use during training.
            schedulers: A list of learning rate schedulers to use during training.
            optimizer_kwargs: Keyword arguments for the optimizer.
            loss_fn_kwargs: Keyword arguments for the loss function.
            callbacks_kwargs: A list of keyword arguments for each callback.
            schedulers_kwargs: A list of keyword arguments for each scheduler.
            device: The device to use. If None, will use GPU if available.
            gradient_clip_val: The value at which to clip gradients.
                Passing ``gradient_clip_val=None`` disables.
            gradient_clip_norm_type: The norm type to use for gradient clipping.
            accumulate_grad_batches: accumulate_grad_batches: Accumulates grads every k batches.
            enable_progress_bar: Whether to enable the progress bar.
        """
        logging.info("Initializing trainer.")
        self.gradient_clip_val: Optional[Union[int, float]] = gradient_clip_val
        self.gradient_clip_norm_type = gradient_clip_norm_type
        self.enable_progress_bar: bool = enable_progress_bar
        self.device: str = device if device is not None else get_device()
        self.accumulate_grad_batches: int = accumulate_grad_batches
        
        self.optimizer: Union[Optimizer, type] = optimizer
        self.optimizer_kwargs: Dict[str, Any] = (optimizer_kwargs if optimizer_kwargs is not None else {})
        
        self.schedulers: List[Union[LRScheduler, type]] = nullable_union_list_object_to_list(schedulers)
        self.schedulers_kwargs: List[Dict[str, Any]] = nullable_union_list_object_to_list(schedulers_kwargs)
        
        self.callbacks: List[Union[Callback, type]] = nullable_union_list_object_to_list(callbacks)
        self.callbacks_kwargs: List[Dict[str, Any]] = nullable_union_list_object_to_list(callbacks_kwargs)
        
        self.loss_fn: Union[_Loss, type] = loss_fn
        self.loss_fn_kwargs: Dict[str, Any] = (loss_fn_kwargs if loss_fn_kwargs is not None else {})
        
        self.metrics = []
        self.metrics_kwargs = []
        self.metrics_history: Dict[str, List[float]]
        
        self.should_stop: bool = False
        self.current_epoch: int
        self.model: Optional[torch.nn.Module] = None
        self.trainable_params: Optional[List[torch.nn.Parameter]] = None

        logging.info("Trainer successfully initialized.")
        
    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Fit one epoch.

        Iterates over all batches in the dataloader (one epoch).
        
        Args:
            dataloader: A :class:`torch.utils.data.DataLoader`
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
            valid_dataloader: A :class:`torch.utils.data.DataLoader`
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
        """Called at the start of fitting.
        
        Args:
            model: The model to fit.
        """
        logging.info("Start fitting the model.")
        
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
        if isinstance(self.loss_fn, type):
            logging.info("Instanciation of loss function %s", self.loss_fn)
            self.loss_fn = self.loss_fn(**self.loss_fn_kwargs)
        self.optimizer.zero_grad()
        
        self._resolve_uninstanciated("callbacks")
        self._resolve_uninstanciated("schedulers")
        self._resolve_uninstanciated("metrics")

        self._call_callbacks("on_fit_start")

    def on_fit_end(self) -> None:
        """Called at the end of fitting."""
        self._call_callbacks("on_fit_end")
        logging.info("Model successfully trained in %s epochs.", self.current_epoch - 1,)
        
    def _call_callbacks(self, call_name: str) -> None:
        """Call all callbacks with the given method name.
        
        Args:
            call_name: The name of the method to call on each callback.
        """
        for callback in self.callbacks:
            fn = getattr(callback, call_name)
            fn(self)
        
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint to `filepath`.
        
        Args:
            filepath: Path to save the checkpoint to.
            
        Raises:
            AttributeError: If no model is attached to the Trainer.
        """
        filepath = resolve_path(filepath)
        if self.model is None:
            raise AttributeError(
                "Saving a checkpoint is only possible if a model is attached "
                "to the Trainer. Did you call `Trainer.save_checkpoint()` "
                "before calling `Trainer.{fit,test}`?"
            )
        self.model.save_state_dict(filepath)
       
    @classmethod
    def from_config(
        cls,
        source: Union[TrainingConfig, str],
    ) -> Any:
        """Construct `Model` instance from `source` configuration.
        
         Args:
            source: A `TrainingConfig` instance or the path to a YAML file
        
        Returns:
            An instance of `Trainer` constructed from `source`.
        """
        if isinstance(source, str):
            source = TrainingConfig.load(source)
        return source._construct_trainer()
