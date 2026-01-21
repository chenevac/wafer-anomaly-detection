import logging
from typing import List, Tuple

from torch.nn import Module

from wafer_ad.model.resnet.resnet import resnet18, resnet34, resnet50, resnext50_32x4d, wide_resnet50_2


class ExtractorFactory:
    @staticmethod
    def build(extractor_name: str) -> Tuple[Module, List]:
        match extractor_name:
            case "resnet18":
                extractor = resnet18(pretrained=True, progress=True)
            case "resnet34":
                extractor = resnet34(pretrained=True, progress=True)
            case "resnet50":
                extractor = resnet50(pretrained=True, progress=True)
            case "resnext50_32x4d":
                extractor = resnext50_32x4d(pretrained=True, progress=True)
            case "wide_resnet50_2":
                extractor = wide_resnet50_2(pretrained=True, progress=True)
            case _:
                raise ValueError(f"Extractor '{extractor_name}' is not supported.")
                
        output_channels = []
        if 'wide' in extractor_name:
            for i in range(3):
                output_channels.append(eval('extractor.layer{}[-1].conv3.out_channels'.format(i+1)))
        else:
            for i in range(3):
                output_channels.append(extractor.eval('layer{}'.format(i+1))[-1].conv2.out_channels)
                
        logging.debug("Channels of extracted features:", output_channels)
        return extractor, output_channels
