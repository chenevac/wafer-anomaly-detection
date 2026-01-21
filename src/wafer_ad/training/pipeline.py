

import logging
from typing import Union
from wafer_ad.data.dataloader import get_data_loaders
from wafer_ad.model.flow import MSFlowModel
from wafer_ad.training.train import Trainer
from wafer_ad.utils.configuration.model_config import ModelConfig
from wafer_ad.utils.configuration.training_config import TrainingConfig


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
        scores = self.trainer.test(self.test_loader)
        for score_name, score_value in scores.items():
            logging.info(f"Test {score_name}: {score_value}")
