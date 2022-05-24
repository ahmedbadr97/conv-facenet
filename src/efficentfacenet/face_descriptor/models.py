from functools import partial

import numpy as np
import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import EfficientNet
from torch import load, save, nn
from torchvision.models.efficientnet import MBConvConfig, model_urls
import torch.nn.functional as F


class FaceDescriptorModel(EfficientNet):

    def __init__(self, download_weights, version, output_size=128, **kwargs):
        progress = True
        args = efficientnet_args(version)
        super(FaceDescriptorModel, self).__init__(*args, **kwargs)
        if download_weights:
            if model_urls.get(version, None) is None:
                raise ValueError(f"No checkpoint is available for model type {version}")
            state_dict = load_state_dict_from_url(model_urls[version], progress=progress)
            self.load_state_dict(state_dict)

        # Change Full connected layer
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.features[-1][0].out_channels, 256),
                                        nn.Dropout(0.3),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 128))

    def load_local_weights(self, path, cuda_weights=False):
        if cuda_weights:
            device = torch.device('cpu')
            state_dict = load(path, map_location=device)
        else:
            state_dict = load(path)
        self.load_state_dict(state_dict)

    def save_weights(self, path):

        state_dict = self.state_dict()
        save(state_dict, path)

    def feature_vector(self, faces, transform=None):
        """
        calculate 128 feature vector for given image(s)

        :param faces: after transform img must be tensor of size 240x240
        :param transform:
        :return: nx128 tensor feature vector where n is images size
        """
        self.eval()
        shape = faces.shape
        if transform is not None:
            faces = transform(faces)
        if len(shape) == 3:
            faces.unsqueeze(0)
        with torch.no_grad():
            output = self(faces)
        return output


class EfficientFacenet(nn.Module):
    def __init__(self, face_features_dim=128, descriptor_weights_path=None):
        super().__init__()

        self.descriptor = FaceDescriptorModel(False, "efficientnet_b1")
        if descriptor_weights_path is not None:
            self.descriptor.load_local_weights(descriptor_weights_path, True)
        self.classifier = nn.Sequential(nn.Linear(face_features_dim * 2, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
                                        nn.Linear(64, 1),nn.Sigmoid())

    def forward(self, face_x, face_y):
        assert face_x.shape == face_y.shape
        assert len(face_x.shape) == len(face_y.shape) == 4
        face_x_features = self.descriptor(face_x)
        face_y_features = self.descriptor(face_y)
        classifier_input = torch.cat((face_x_features, face_y_features), dim=1)
        output = self.classifier(classifier_input)
        return output

    def load_local_weights(self, path, cuda_weights=False):
        if cuda_weights:
            device = torch.device('cpu')
            state_dict = load(path, map_location=device)
        else:
            state_dict = load(path)
        self.load_state_dict(state_dict)

    def save_weights(self, path):

        state_dict = self.state_dict()
        save(state_dict, path)

    def identify_faces(self, face_x, face_y, transform=None):
        if transform is not None:
            face_x = transform(face_x)
            face_y = transform(face_y)
        self.eval()
        with torch.no_grad():
            output = self.forward(face_x, face_y)
        return output

    def classify_face_features(self, face_x, face_y):
        self.eval()
        if not isinstance(face_x,torch.Tensor):
            face_x=torch.tensor(face_x)
        if not isinstance(face_y,torch.Tensor):
            face_y=torch.tensor(face_y)

        if len(face_x.shape)==1:
            face_x=face_x.unsqueeze(0)
        if len(face_y.shape)==1:
            face_y=face_y.unsqueeze(0)
        with torch.no_grad():
            classifier_input = torch.cat((face_x, face_y), dim=1)
            output = self.classifier(classifier_input)
        return output[0].numpy()


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


def efficientnet_args(version):
    if version == "efficientnet_b1":
        width_mult = 1.0
        depth_mult = 1.1
        dropout = 0.2
    elif version == "efficientnet_b0":
        width_mult = 1.0
        depth_mult = 1.0
        dropout = 0.2
    elif version == "efficientnet_b4":
        width_mult = 1.4
        depth_mult = 1.8
        dropout = 0.4
    else:
        raise ValueError(f"invalid version of efficientnet '{version}' ")
    return get_inverted_residual_setting(width_mult, depth_mult), dropout
