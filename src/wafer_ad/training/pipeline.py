

import logging
from typing import Union

from wafer_ad.data.dataloader import get_data_loaders
from wafer_ad.model.flow import MSFlowModel
from wafer_ad.training.trainer import Trainer
from wafer_ad.utils.configuration.model_config import ModelConfig
from wafer_ad.utils.configuration.training_config import TrainingConfig
from wafer_ad.inference.test import test
from wafer_ad.utils.figure import display_roc


class TrainingPipeline:
    """Pipeline to train and test the MSFlow model."""
    def __init__(
        self,
        train_config: Union[str, TrainingConfig],
        model_config: Union[str, ModelConfig],
    ) -> None:
        """Initialize the training pipeline.
        
        Args:
            train_config: Configuration for training, either as a file path or a TrainingConfig object.
            model_config: Configuration for the model, either as a file path or a ModelConfig object.
        """
        self.train_config = TrainingConfig.load(train_config) if isinstance(train_config, str) else train_config
        self.model_config = ModelConfig.load(model_config) if isinstance(model_config, str) else model_config
        
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            batch_size=self.train_config.batch_size,
            data_folder=self.train_config.data_folder,
        )
        
        
        self.trainer = Trainer.from_config(self.train_config)
        self.model = MSFlowModel.from_config(self.model_config)
        
    def run(self) -> None:
        """Run the training and testing pipeline."""
        self.trainer.train(
            model=self.model,
            train_dataloader=self.train_loader,
            valid_dataloader=self.val_loader,
            n_epochs=self.train_config.n_epochs,
        )
        metrics = test(
            model=self.model,
            dataloader=self.test_loader,
            enable_progress_bar=True,
            device=self.trainer.device,
        )
        display_roc(
            metrics["imagewise_retrieval_metrics"]["fpr"],
            metrics["imagewise_retrieval_metrics"]["tpr"],
            title="Image-wise ROC Curve"
        )
        print("Image-wise AUROC:", metrics["imagewise_retrieval_metrics"]["auroc"])
        
        display_roc(
            metrics["pixelwise_retrieval_metrics_add"]["fpr"],
            metrics["pixelwise_retrieval_metrics_add"]["tpr"],
            title="Pixel-wise ROC Curve (Additive)",
        )
        print("Pixel-wise AUROC (Additive):", metrics["pixelwise_retrieval_metrics_add"]["auroc"])
        display_roc(
            metrics["pixelwise_retrieval_metrics_mul"]["fpr"],
            metrics["pixelwise_retrieval_metrics_mul"]["tpr"],
            title="Pixel-wise ROC Curve (Multiplicative)",
        )
        print("Pixel-wise AUROC (Multiplicative):", metrics["pixelwise_retrieval_metrics_mul"]["auroc"])