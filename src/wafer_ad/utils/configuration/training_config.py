from dataclasses import dataclass
from typing import Any, Dict

from wafer_ad.utils.configuration.base import BaseConfig
from wafer_ad.utils.parsing import get_all_wafer_ad_classes, traverse_and_apply


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for all trainings."""

    # Fields
    class_name: str
    arguments: Dict[str, Any]
    n_epochs: int
    batch_size: int
    data_folder: str

    def _construct_trainer(
        self,
    ) -> "Trainer":
        import wafer_ad.training

        namespace_classes = get_all_wafer_ad_classes(
            wafer_ad.training
        )

        arguments = dict(**self.arguments)
        arguments = traverse_and_apply(arguments, self._deserialise)

        return namespace_classes[self.class_name](**arguments)

    @classmethod
    def _deserialise(
        cls, obj: Any
    ) -> Any:
        if isinstance(obj, TrainingConfig):
            from wafer_ad.training.trainer import Trainer

            return Trainer.from_config(obj)
        elif isinstance(obj, str) and obj.startswith("!class"):
            module, class_name = obj.split()[1:]
            exec(f"from {module} import {class_name}")
            return eval(class_name)
        else:
            return obj