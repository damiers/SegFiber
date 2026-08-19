import functools

import torch
import torch.nn as nn

from ..core.registry import ARCH_REGISTRY


def get_norm_layer(norm_type="instance", dim=3):
    batch_norm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
    instance_norm = nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d
    if norm_type == "batch":
        return functools.partial(batch_norm, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(instance_norm, affine=False, track_running_stats=False)
    if norm_type == "identity":
        return lambda _channels: nn.Identity()
    raise ValueError(f"Unknown normalization: {norm_type}")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm_type="batch", dim=3):
        super().__init__()
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        norm_layer = get_norm_layer(norm_type, dim)
        use_bias = norm_type == "instance"
        self.conv = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            conv(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.conv(inputs)


@ARCH_REGISTRY.register("unet")
class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=(32, 64, 128),
        *,
        norm_type="batch",
        dim=3,
    ):
        super().__init__()
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        conv_transpose = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
        self.MaxPool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for feature in features:
            self.downs.append(
                DoubleConv(in_channels, feature, norm_type=norm_type, dim=dim)
            )
            in_channels = feature

        for feature in reversed(features[:-1]):
            self.ups.append(
                conv_transpose(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(
                DoubleConv(feature * 2, feature, norm_type=norm_type, dim=dim)
            )

        self.final_conv = nn.Sequential(
            conv(features[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    @classmethod
    def init_from_config(cls, params, config):
        return cls(**params)

    def forward(self, inputs):
        skip_connections = []
        x = inputs
        for index, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            if index != len(self.downs) - 1:
                x = self.MaxPool(kernel_size=2)(x)

        skip_connections.reverse()
        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)
            skip = skip_connections[index // 2 + 1]
            if x.shape != skip.shape:
                padding = []
                for current, target in zip(reversed(x.shape[2:]), reversed(skip.shape[2:])):
                    padding.extend((0, target - current))
                x = nn.functional.pad(x, padding)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[index + 1](x)
        return self.final_conv(x)
