import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_pool(*args, **kwargs):
    """Configurable Conv block with Batchnorm and Dropout"""
    return nn.Sequential(
        nn.Conv3d(*args, **kwargs),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2),
    )

params = [
    {
        "in_channels": 1,
        "kernel_size": 11,
        "out_channels": 144,
        "stride": 3,
    },
    {
        "in_channels": 144,
        "kernel_size": 5,
        "out_channels": 192,
        "stride": 2,
        "bias": False,
    },
    {
        "in_channels": 192,
        "kernel_size": 5,
        "out_channels": 192,
        "stride": 1,
        "bias": False,
    }
]

class BMENet(nn.Module):
    """Configurable Net from https://www.frontiersin.org/articles/10.3389/fneur.2020.00244/full"""

    def __init__(self, n_classes):
        """Init"""

        super(BMENet, self).__init__()
        layers = [conv_pool(**block_kwargs) for block_kwargs in params]
        layers.append(nn.Dropout3d(.4))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=192, out_features=374))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(.4))
        layers.append(nn.Linear(in_features=374, out_features=374))
        layers.append(nn.Linear(in_features=374, out_features=n_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x

#BELOW ARE ORIGINAL PARAMETERS
"""
def conv_pool(*args, **kwargs):
    return nn.Sequential(
        nn.Conv3d(*args, **kwargs),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2),
    )

params = [
    {
        "in_channels": 1,
        "kernel_size": 11,
        "out_channels": 144,
        "stride": 3,
    },
    {
        "in_channels": 144,
        "kernel_size": 5,
        "out_channels": 192,
        "stride": 2,
        "bias": False,
    },
    {
        "in_channels": 192,
        "kernel_size": 5,
        "out_channels": 192,
        "stride": 1,
        "bias": False,
    },
]
"""