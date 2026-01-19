from typing import List
import torch
from torch import nn
from torch.nn.modules.loss import _Loss

class MSFlowLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        z_list: List[torch.Tensor],
        jac: torch.Tensor,
    ) -> torch.Tensor:
        loss = sum(
            0.5 * torch.sum(z ** 2, dim=(1, 2, 3))
            for z in z_list
        )  # normality term
        loss = loss - jac  # jacobian adjustment
        return loss.mean()
