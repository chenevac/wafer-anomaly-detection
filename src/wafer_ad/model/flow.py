import logging
import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from wafer_ad.model.FrEIA import FusionCouplingLayer
from wafer_ad.model.extractor import ExtractorFactory
from wafer_ad.utils.configuration.configurable import Configurable
from wafer_ad.utils.configuration.model_config import ModelConfig




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
        seed: Optional[int] = 1,
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
            nodes.append(Ff.Node(nodes[-n_inputs], Fm.PermuteRandom, {"seed": seed+idx}, name='permute_{}'.format(idx)))
        nodes.append(Ff.Node([(nodes[-n_inputs+i], 0) for i in range(n_inputs)], FusionCouplingLayer, {'clamp': clamp_alpha}, name='fusion flow'))
        for idx, c_feat in enumerate(c_feats):
            nodes.append(Ff.OutputNode(eval('nodes[-idx-1].out{}'.format(idx)), name='output_{}'.format(idx)))
        fusion_flow = Ff.GraphINN(nodes)

        return parallel_flows, fusion_flow


class MSFlowModel(nn.Module, Configurable):
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
        """
        Post-traitement des sorties du forward pour calculer les scores d'anomalie.
        
        Args:
            z_list: Liste de tenseurs latents de différentes échelles issus du forward
            jac: Jacobien total (non utilisé ici mais gardé pour compatibilité)
        
        Returns:
            anomaly_score: Score d'anomalie moyen par image (numpy array)
            anomaly_score_map_add: Carte d'anomalie avec fusion additive (numpy array)
            anomaly_score_map_mul: Carte d'anomalie avec fusion multiplicative (numpy array)
        """
        # Récupère la taille du batch depuis le premier tenseur de la liste
        B = z_list[0].shape[0]

        # Initialisation des listes pour stocker les cartes de log-probabilité et de probabilité
        logp_maps = []
        prop_maps = []

        # Traitement de chaque échelle
        for z in z_list:
            # Calcul des outputs : log-densité gaussienne simplifiée
            # -0.5 * ||z||^2 correspond à log(exp(-0.5 * ||z||^2)) dans une gaussienne standard
            # mean(z**2, dim=1) fait la moyenne sur le canal pour obtenir une carte 2D (B, H, W)
            outputs = -0.5 * torch.mean(z**2, dim=1)

            # ===== Carte de log-probabilité =====
            # Redimensionne la carte à la taille d'entrée de l'image
            # unsqueeze(1) ajoute une dimension canal pour interpolate : (B, H, W) -> (B, 1, H, W)
            # squeeze(1) retire cette dimension après interpolation : (B, 1, H', W') -> (B, H', W')
            logp_maps.append(
                torch.nn.functional.interpolate(
                    outputs.unsqueeze(1),
                    size=self.img_size,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)
            )

            # ===== Carte de probabilité =====
            # Normalisation : soustrait le maximum pour éviter l'overflow dans exp()
            # max(-1, keepdim=True)[0] trouve le max sur la dimension W
            # max(-2, keepdim=True)[0] trouve ensuite le max sur la dimension H
            # Résultat : le max global de chaque image du batch
            outputs_norm = outputs - outputs.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
            
            # Conversion en probabilités : exp(log_prob) donne des valeurs dans [0, 1]
            prob_map = torch.exp(outputs_norm)

            # Redimensionne la carte de probabilité à la taille d'entrée
            prop_maps.append(
                torch.nn.functional.interpolate(
                    prob_map.unsqueeze(1),
                    size=self.img_size,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)
            )

        # =============================
        # Fusion multiplicative
        # =============================
        # Somme des log-probabilités = produit des probabilités (log(a*b) = log(a) + log(b))
        logp_map = sum(logp_maps)
        
        # Normalisation de la carte de log-probabilité fusionnée
        # Soustrait le max pour stabilité numérique
        logp_map -= logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]

        # Conversion en probabilités : exp(sum(log_prob)) = produit des probabilités
        prop_map_mul = torch.exp(logp_map)
        
        # Calcul de la carte d'anomalie multiplicative
        # Les zones normales ont une haute probabilité -> faible score d'anomalie
        # Les zones anormales ont une faible probabilité -> score d'anomalie élevé
        # On fait : max - valeur_actuelle pour inverser
        anomaly_score_map_mul = (
            prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
        )

        # Calcul du nombre de pixels à considérer pour le top-k
        # top_k est un pourcentage (ex: 0.03 = 3% des pixels)
        top_k = int(self.img_size[0] * self.img_size[1] * self.top_k)

        # Calcul du score d'anomalie final par image
        # reshape(B, -1) : aplatit la carte 2D en vecteur 1D
        # topk(top_k, dim=-1)[0] : sélectionne les top_k scores les plus élevés
        # mean(dim=1) : moyenne des top_k scores pour obtenir un score par image
        # detach().cpu().numpy() : convertit en numpy array pour compatibilité
        anomaly_score = (
            anomaly_score_map_mul
            .reshape(B, -1)
            .topk(top_k, dim=-1)[0]
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        # =============================
        # Fusion additive
        # =============================
        # Somme directe des cartes de probabilité (fusion additive)
        # Conversion en numpy pour cohérence avec la fonction originale
        prop_map_add = sum(prop_maps).detach().cpu().numpy()
        
        # Calcul de la carte d'anomalie additive
        # max(axis=(1, 2), keepdims=True) : trouve le max sur H et W, garde les dimensions
        # Soustraction pour obtenir le score d'anomalie (max - valeur)
        anomaly_score_map_add = (
            prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add
        )

        # Retour des résultats
        # anomaly_score : (B,) score moyen par image
        # anomaly_score_map_add : (B, H, W) carte d'anomalie fusion additive
        # anomaly_score_map_mul : (B, H, W) carte d'anomalie fusion multiplicative
        return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul.detach().cpu().numpy()
            
    
    def save_state_dict(self, path: str) -> None:
        if not path.endswith(".pth"):
            logging.warning("It is recommended to use the .pth suffix for state_dict files.")
        torch.save(self.state_dict(), path)
        logging.info("Model state_dict saved to %s", path)


    def load_state_dict(self, path: str, strict: bool = True) -> None:
        logging.info("Loading weights from %s", path)
        state_dict = torch.load(path, map_location="cpu")
        
        # Réorganiser les poids des PermuteFixed dans fusion_flow
        # FrEIA peut changer l'ordre des modules entre instanciations
        permute_keys = [k for k in state_dict.keys() if 'fusion_flow.module_list' in k and '.perm' in k]
        
        if permute_keys:
            # Extraire les poids de permutation du checkpoint
            saved_perms = {}
            for key in permute_keys:
                if key.endswith('.perm') or key.endswith('.perm_inv'):
                    saved_perms[key] = state_dict[key]
            
            # Trouver les correspondances basées sur la taille des tenseurs
            current_perms = {}
            for name, param in self.named_parameters():
                if 'fusion_flow.module_list' in name and ('.perm' in name):
                    current_perms[name] = param
            
            # Créer un mapping taille -> clés
            saved_by_size = {}
            for key, tensor in saved_perms.items():
                size = tensor.shape[0]
                param_type = 'perm' if key.endswith('.perm') else 'perm_inv'
                if size not in saved_by_size:
                    saved_by_size[size] = {}
                saved_by_size[size][param_type] = (key, tensor)
            
            # Remapper les poids selon la taille
            for current_key, current_param in current_perms.items():
                size = current_param.shape[0]
                param_type = 'perm' if current_key.endswith('.perm') else 'perm_inv'
                
                if size in saved_by_size and param_type in saved_by_size[size]:
                    saved_key, saved_tensor = saved_by_size[size][param_type]
                    state_dict[current_key] = saved_tensor
                    # Marquer comme utilisé
                    del saved_by_size[size][param_type]
                    if not saved_by_size[size]:
                        del saved_by_size[size]
        
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_config(
        cls,
        source: Union[ModelConfig, str],
    ) -> Any:
        """Construct `Model` instance from `source` configuration."""
        if isinstance(source, str):
            source = ModelConfig.load(source)
        return source._construct_model()
        
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