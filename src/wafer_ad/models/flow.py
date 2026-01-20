import logging
import math
from typing import List, Tuple, Union

import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from wafer_ad.models.FrEIA import FusionCouplingLayer
from wafer_ad.models.extractor import ExtractorFactory
from wafer_ad.utils.config import Config




def subnet_conv(dims_in, dims_out):
    return nn.Sequential(nn.Conv2d(dims_in, dims_in, 3, 1, 1), nn.ReLU(True), nn.Conv2d(dims_in, dims_out, 3, 1, 1))

def subnet_conv_bn(dims_in, dims_out):
    return nn.Sequential(nn.Conv2d(dims_in, dims_in, 3, 1, 1), nn.BatchNorm2d(dims_in), nn.ReLU(True), nn.Conv2d(dims_in, dims_out, 3, 1, 1))

class subnet_conv_ln(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_mid = dim_in
        self.conv1 = nn.Conv2d(dim_in, dim_mid, 3, 1, 1)
        self.ln = nn.LayerNorm(dim_mid)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(dim_mid, dim_out, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.ln(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.conv2(out)

        return out

def single_parallel_flows(c_feat, c_cond, n_block, clamp_alpha, subnet=subnet_conv_ln):
    flows = Ff.SequenceINN(c_feat, 1, 1)
    print('Build parallel flows: channels:{}, block:{}, cond:{}'.format(c_feat, n_block, c_cond))
    for k in range(n_block):
        flows.append(Fm.AllInOneBlock, cond=0, cond_shape=(c_cond, 1, 1), subnet_constructor=subnet, affine_clamping=clamp_alpha,
            global_affine_type='SOFTPLUS')
    return flows


def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


class MSFlowFactory:
    @staticmethod
    def build(
        c_feats,
        c_conds: List[int],  # [64, 64, 64]
        n_blocks: List[int],  # [2, 5, 8]
        clamp_alpha: float = 1.9,
    ) -> Tuple[List[Ff.SequenceINN], Ff.GraphINN]:
        parallel_flows = []
        for c_feat, c_cond, n_block in zip(c_feats, c_conds, n_blocks):
            parallel_flows.append(
                single_parallel_flows(c_feat, c_cond, n_block, clamp_alpha, subnet=subnet_conv_ln))
        
        print("Build fusion flow with channels", c_feats)
        nodes = list()
        n_inputs = len(c_feats)
        for idx, c_feat in enumerate(c_feats):
            nodes.append(Ff.InputNode(c_feat, 1, 1, name='input{}'.format(idx)))
        for idx in range(n_inputs):
            nodes.append(Ff.Node(nodes[-n_inputs], Fm.PermuteRandom, {}, name='permute_{}'.format(idx)))
        nodes.append(Ff.Node([(nodes[-n_inputs+i], 0) for i in range(n_inputs)], FusionCouplingLayer, {'clamp': clamp_alpha}, name='fusion flow'))
        for idx, c_feat in enumerate(c_feats):
            nodes.append(Ff.OutputNode(eval('nodes[-idx-1].out{}'.format(idx)), name='output_{}'.format(idx)))
        fusion_flow = Ff.GraphINN(nodes)

        return parallel_flows, fusion_flow




class MSFlowModel(nn.Module):
    def __init__(
        self,
        c_conds, n_blocks,
        extractor_name: str = "wide_resnet50_2",
        pool_type: str = "avg",
        top_k: float = 0.03,
        img_size: Tuple[int, int] = (512, 512),
    ) -> None:
        super().__init__()

        self.pool_type = pool_type
        self.c_conds = c_conds
        self.top_k = top_k
        self.img_size = img_size

        self.extractor, self.output_channels = ExtractorFactory.build(extractor_name)
        parallel_flows, self.fusion_flow = MSFlowFactory.build(self.output_channels, c_conds=c_conds, n_blocks=n_blocks)
        self.parallel_flows = nn.ModuleList(parallel_flows)

        self.extractor.eval()  # MSFlow: extractor gelé

    def _get_pool(self):
        if self.pool_type == "avg":
            return nn.AvgPool2d(3, 2, 1)
        elif self.pool_type == "max":
            return nn.MaxPool2d(3, 2, 1)
        else:
            return nn.Identity()

    def forward(self, image):
        h_list = self.extractor(image)
        pool = self._get_pool()

        z_list = []
        jac_list = []

        for h, flow, c_cond in zip(h_list, self.parallel_flows, self.c_conds):
            y = pool(h)
            B, _, H, W = y.shape

            cond = positionalencoding2d(c_cond, H, W)#.to(y.device)
            cond = cond.type_as(y).unsqueeze(0).repeat(B, 1, 1, 1)

            z, jac = flow(y, [cond])
            z_list.append(z)
            jac_list.append(jac)

        z_list, fuse_jac = self.fusion_flow(z_list)
        total_jac = fuse_jac + sum(jac_list)

        return z_list, total_jac
    
    def post_process_forward(self, z_list, jac):
        B = z_list[0].shape[0]

        logp_maps = []
        prop_maps = []

        for z in z_list:
            # (B, H, W)
            outputs = -0.5 * torch.mean(z**2, dim=1)

            # Log-probability map
            logp_maps.append(
                torch.nn.functional.interpolate(
                    outputs.unsqueeze(1),
                    size=self.img_size,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)
            )

            # Probability map
            outputs_norm = outputs - outputs.amax(dim=(-2, -1), keepdim=True)
            prob_map = torch.exp(outputs_norm)

            prop_maps.append(
                torch.nn.functional.interpolate(
                    prob_map.unsqueeze(1),
                    size=self.img_size,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)
            )

        # -----------------------------
        # Multiplicative fusion
        # -----------------------------
        logp_map = sum(logp_maps)
        logp_map -= logp_map.amax(dim=(-2, -1), keepdim=True)

        prop_map_mul = torch.exp(logp_map)
        anomaly_score_map_mul = (
            prop_map_mul.amax(dim=(-2, -1), keepdim=True) - prop_map_mul
        )

        top_k = int(self.img_size[0] * self.img_size[1] * self.top_k)

        anomaly_score = (
            anomaly_score_map_mul
            .reshape(B, -1)
            .topk(top_k, dim=-1)[0]
            .mean(dim=1)
        )

        # -----------------------------
        # Additive fusion
        # -----------------------------
        prop_map_add = sum(prop_maps)
        anomaly_score_map_add = (
            prop_map_add.amax(dim=(-2, -1), keepdim=True) - prop_map_add
        )

        return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul
            
    
    def save_state_dict(self, path: str) -> None:
        if not path.endswith(".pth"):
            logging.warning("It is recommended to use the .pth suffix for state_dict files.")
        torch.save(self.state_dict(), path)
        logging.info("Model state_dict saved to %s", path)


    def load_weights(self, path: str, strict: bool = True) -> None:
        logging.info("Loading weights from %s", path)
        state_dict = torch.load(path, map_location="cpu")
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_config(
        cls,
        source: Union[Config, str],
    ) -> "MSFlowModel":
        """Create MSFlowModel from config file.

        Args:
            source: Either a Config object or a path to a config file.
        """
        if isinstance(source, str):
            logging.info("Loading MSFlowModel config from %s", source)
            config = Config.from_yaml(source)
        else:
            config = source

        logging.info("Creating MSFlowModel from config.")
        return cls(
            c_conds=config.c_conds,
            n_blocks=config.n_blocks,
            extractor_name=config.extractor_name,
            pool_type=config.pool_type,
        )
        
    def train(self, mode=True):
        super().train(mode)  # <-- met à jour self.training et tous les sous-modules par défaut
        # override spécifique pour l'extractor
        self.extractor.eval()  # freeze extractor, pas de BN/Dropout mis à jour
        for param in self.extractor.parameters():
            param.requires_grad = False
        # flows restent en mode train/val selon 'mode'
        for flow in self.parallel_flows:
            flow.train(mode)
        self.fusion_flow.train(mode)
        return self
    
    def eval(self):
        return self.train(False)