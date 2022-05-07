from functools import partial

from torch.hub import load_state_dict_from_url
from torchvision.models import EfficientNet
from torch import nn, load, save
from torchvision.models.efficientnet import MBConvConfig, model_urls


class FaceDescriptorModel(EfficientNet):

    def __init__(self, download_weights, version, **kwargs):
        progress = True
        super(FaceDescriptorModel, self).__init__(efficientnet_args(version, kwargs))
        if download_weights:
            if model_urls.get(version, None) is None:
                raise ValueError(f"No checkpoint is available for model type {version}")
            state_dict = load_state_dict_from_url(model_urls[version], progress=progress)
            self.load_state_dict(state_dict)

        # Change Full connected layer
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(2048, 128))

    def load_local_weights(self, path):
        state_dict = load(path)
        self.load_state_dict(state_dict)

    def save_weights(self, path):
        state_dict = self.state_dict()
        save(state_dict, path)


def get_inverted_residual_setting(width_mult, depth_mult):
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1)]
    return inverted_residual_setting


def efficientnet_args(version, **kwargs):
    if version == "efficientnet_b1":
        width_mult = 1.0
        depth_mult = 1.1
        dropout = 0.2
    elif version == "efficientnet_b0":
        width_mult = 0.1
        depth_mult = 1.0
        dropout = 0.2
    else:
        raise ValueError(f"invalid version of efficientnet '{version}' ")
    return get_inverted_residual_setting(width_mult, depth_mult), dropout, kwargs
