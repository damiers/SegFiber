import torch
import torch.nn as nn
import functools

# ==========
# normaliz layer
# ==========
def get_norm_layer(norm_type='instance', dim=3):
    if dim == 2:
        BatchNorm = nn.BatchNorm2d
        InstanceNorm = nn.InstanceNorm2d
    elif dim == 3:
        BatchNorm = nn.BatchNorm3d
        InstanceNorm = nn.InstanceNorm3d
    else:
        raise Exception('Invalid dim.')
    
    if norm_type == 'batch':
        norm_layer = functools.partial(BatchNorm, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(InstanceNorm, affine=False, track_running_stats=False)
    elif norm_type == 'identity':
        def norm_layer(x):
            return lambda t:t
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# ==========
# UNet
# ==========
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm_type='batch', dim=3):
        super(DoubleConv, self).__init__()

        if dim == 2:
            Conv = nn.Conv2d
        elif dim == 3:
            Conv = nn.Conv3d
        else:
            raise Exception('Invalid dim.')
        
        norm_layer=get_norm_layer(norm_type, dim=dim)
        use_bias = True if norm_type=='instance' else False

        self.conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128], *, norm_type='batch', dim=3):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        if dim == 2:
            Conv = nn.Conv2d
            ConvTranspose = nn.ConvTranspose2d
            self.MaxPool = nn.MaxPool2d
        elif dim == 3:
            Conv = nn.Conv3d
            ConvTranspose = nn.ConvTranspose3d
            self.MaxPool = nn.MaxPool3d
        else:
            raise Exception('Invalid dim.')

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, norm_type=norm_type, dim=dim))
            in_channels = feature

        # Decoder
        for feature in reversed(features[:-1]):
            self.ups.append(
                ConvTranspose(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature, norm_type=norm_type, dim=dim))

        self.final_conv = nn.Sequential(
            Conv(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_connections = []

        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            if i != len(self.downs)-1:
                x = self.MaxPool(kernel_size=2)(x)

        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i//2+1]
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)

        x = self.final_conv(x)
        return x

def get_model(args):
    model = UNet(in_channels=args.in_channels,
                 out_channels=args.out_channels,
                 features=args.features,
                 norm_type=args.norm_type,
                 dim=3)
    return model

if __name__ == '__main__':
    import torch
    from torchinfo import summary
    from argparse import Namespace
    pretrain_ckpt_path = '/home/ryuuyou/E5/project/SegFiber_dev/eval/model/universal_tiny.pth'
    ckpt = torch.load(pretrain_ckpt_path, map_location='cpu')
    
    args = Namespace()
    args.in_channels = 1
    args.out_channels = 1
    args.features = [32,64,128]
    args.norm_type = 'batch'
    model = get_model(args)

    model.load_state_dict(ckpt)
