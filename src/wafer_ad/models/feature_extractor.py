from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.nn import functional as F

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        img_size,
        n_scales: int = 3,
    ):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        self.img_size = img_size
        self.n_scales = n_scales

    def eff_ext(self, x, use_layer=35):
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == use_layer:
                return x

    def forward(self, x):
        y = list()
        for s in range(self.n_scales):
            feat_s = F.interpolate(x, size=(self.img_size[0] // (2 ** s), self.img_size[1] // (2 ** s))) if s > 0 else x
            feat_s = self.eff_ext(feat_s)

            y.append(feat_s)
        return y
    
    