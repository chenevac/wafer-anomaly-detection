import logging
from math import exp
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
from torch import nn

from wafer_ad.models.FrEIA import CrossConvolutions, InputNode, Node, OutputNode, ParallelPermute, ReversibleGraphNet, parallel_glow_coupling_layer
from wafer_ad.utils.config import Config
from wafer_ad.utils.devices import get_device
from wafer_ad.models.feature_extractor import FeatureExtractor


class CSFlow(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        n_coupling_blocks: int = 4,
        input_dim: int = 304,
        clamp: float = 3.0,
        fc_internal: int = 1024,
        n_scales: int = 3,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.n_coupling_blocks = n_coupling_blocks
        self.input_dim = input_dim
        self.clamp = clamp
        self.fc_internal = fc_internal
        self.n_scales = n_scales

        self.nf = self._build_flow()
        self.fe = FeatureExtractor(img_size=(512,512), n_scales=self.n_scales)
        
    def _build_flow(self) -> ReversibleGraphNet:
        map_size = (
            self.img_size[0] // 32,
            self.img_size[1] // 32,
        )

        nodes: List = []
        
    
        nodes.append(InputNode(self.input_dim, map_size[0], map_size[1], name='input'))
        nodes.append(InputNode(self.input_dim, map_size[0] // 2, map_size[1] // 2, name='input2'))
        nodes.append(InputNode(self.input_dim, map_size[0] // 4, map_size[1] // 4, name='input3'))

        for k in range(self.n_coupling_blocks):
            if k == 0:
                node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
            else:
                node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]

            nodes.append(Node(node_to_permute, ParallelPermute, {'seed': k}, name=F'permute_{k}'))
            nodes.append(Node([nodes[-1].out0, nodes[-1].out1, nodes[-1].out2], parallel_glow_coupling_layer,
                            {'clamp': self.clamp, 'F_class': CrossConvolutions,
                            'F_args': {'channels_hidden': self.fc_internal,
                                        'kernel_size': ([3] * (self.n_coupling_blocks - 1) + [5])[k], 'block_no': k}},
                            name=F'fc1_{k}'))

        nodes.append(OutputNode([nodes[-1].out0], name='output_end0'))
        nodes.append(OutputNode([nodes[-2].out1], name='output_end1'))
        nodes.append(OutputNode([nodes[-3].out2], name='output_end2'))
        nf = ReversibleGraphNet(nodes, n_jac=3)
        return nf
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.fe(x)
        return self.nf(features), self.nf.jacobian(run_forward=False)
    
    def save(self, path: str) -> None:
        """Save entire model to `path`."""
        if not path.endswith(".pth"):
            logging.warning(
                "It is recommended to use the .pth suffix for model files."
            )
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self, path)
        logging.info("Model saved to %s", path)
        
    
    @classmethod
    def load(cls, path: str) -> "CSFlow":
        """Load entire model from `path`."""
        logging.info("Loading model from %s", path)
        return torch.load(path)

    def save_state_dict(self, path: str) -> None:
        """Save model `state_dict` to `path`."""
        if not path.endswith(".pth"):
            logging.warning(
                "It is recommended to use the .pth suffix for "
                "state_dict files."
            )
        torch.save(self.state_dict(), path)
        logging.info("Model state_dict saved to %s", path)

    def load_state_dict(self, path: Union[str, Dict], **kargs: Optional[Any]) -> "CSFlow":
        """Load model `state_dict` from `path`."""
        logging.info("Loading weights from %s", path)
        if isinstance(path, str):
            state_dict = torch.load(
                path,
                map_location=torch.device(get_device(verbose=False))
            )
        else:
            state_dict = path
        return super().load_state_dict(state_dict, **kargs)
    
    @classmethod
    def from_config(
        cls,
        source: Union[Config, str],
    ) -> Any:
        pass