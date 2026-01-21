"""Config classes for the `onlinelearning.models` module."""


from dataclasses import dataclass
from typing import Any, Dict

from wafer_ad.utils.configuration.base import BaseConfig
from wafer_ad.utils.parsing import get_all_wafer_ad_classes, traverse_and_apply


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for all `Model`s."""

    # Fields
    class_name: str
    arguments: Dict[str, Any]

    def _construct_model(self) -> "MSFlowModel":
        import wafer_ad.model

        namespace_classes = get_all_wafer_ad_classes(
            wafer_ad.model
        )

        arguments = dict(**self.arguments)
        arguments = traverse_and_apply(
            arguments,
            self._deserialise
        )

        return namespace_classes[self.class_name](**arguments)

    @classmethod
    def _deserialise(cls, obj: Any) -> Any:
        if isinstance(obj, ModelConfig):
            from wafer_ad.model.flow import MSFlowModel
            return MSFlowModel.from_config(obj)
        elif isinstance(obj, str) and obj.startswith("!class"):
            module, class_name = obj.split()[1:]
            exec(f"from {module} import {class_name}")
            return eval(class_name)
        else:
            return obj
