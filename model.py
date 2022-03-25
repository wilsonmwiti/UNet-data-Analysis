from matplotlib.cbook import normalize_kwargs
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormNonlinear(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_kwargs=None, 
                 norm_kwargs=None,
                 nonlinear_kwargs=None):
        
        super(ConvNormNonlinear, self).__init__()
        if nonlinear_kwargs is None:
            nonlinear_kwargs = {'negative_slope': 1e-2}
        if norm_kwargs is None:
            norm_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        
        self.conv = nn.Conv2d(input_channels, output_channels, **conv_kwargs)
        self.norm = nn.InstanceNorm2d(output_channels, **norm_kwargs)
        self.nonlinear = nn.LeakyReLU(**nonlinear_kwargs)
        
    def forward(self, x):
        x = self.conv(x)
        return self.nonlinear(self.norm(x))

class DoubleConvNormNonlinear(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_kwargs=None, 
                 norm_kwargs=None,
                 nonlinear_kwargs=None):
        
        super(DoubleConvNormNonlinear, self).__init__()
        
        self.conv_norm_nonlinear1 = ConvNormNonlinear(input_channels, output_channels, conv_kwargs, norm_kwargs,nonlinear_kwargs)
        self.conv_norm_nonlinear2 = ConvNormNonlinear(output_channels, output_channels, conv_kwargs, norm_kwargs,nonlinear_kwargs)        
        
    def forward(self, x):
        x = self.conv_norm_nonlinear1(x)
        x = self.conv_norm_nonlinear2(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(Upsample, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
    
class UNet2D(nn.Module):
    def __init__(self, config = None):
        super(UNet2D, self).__init__()
        if config == None:
            config = {'depth' : 4, 
                      'down_block_channels': [
                          (1, 64),
                          (64, 128),
                          (128, 256),
                          (256, 512)
                      ],
                      'up_block_channels': [
                          (128, 64),
                          (256, 128),
                          (512, 256),
                          (1024, 512),
                      ],
                      'conv_trans_channels': [
                          (128, 64),
                          (256, 128),
                          (512, 256),
                          (1024, 512),
                      ],
                      'bottom_block_channels': (512, 1024),
                      'output_conv_channels': (64, 1)
                      }
        self.depth = config['depth']
        down_block_channels = config['down_block_channels']
        up_block_channels = config['up_block_channels']
        bottom_block_channels = config['bottom_block_channels']
        conv_trans_channels = config['conv_trans_channels']
        output_conv_channels = config['output_conv_channels']
        
        self.downsample = nn.ModuleList([nn.MaxPool2d((2,2)) for _ in range(self.depth)])
        # self.upsample = [Upsample(scale_factor=2) for _ in range(self.depth)]
        
        self.down_blocks = nn.ModuleList([DoubleConvNormNonlinear(channels[0], channels[1]) for channels in down_block_channels])
        self.bottom_block = DoubleConvNormNonlinear(bottom_block_channels[0], bottom_block_channels[1])
        self.up_blocks = nn.ModuleList([DoubleConvNormNonlinear(channels[0], channels[1]) for channels in up_block_channels])
        self.conv_trans = nn.ModuleList([nn.ConvTranspose2d(channels[0], channels[1], kernel_size=2, stride=2) for channels in conv_trans_channels])
        self.output_conv = nn.Conv2d(output_conv_channels[0], output_conv_channels[1], kernel_size=3, stride=1, padding=1)
        # self.output_convs = [nn.Conv2d() for _ in range(depth)]

    def forward(self, x):
        skips = []
        for d in range(self.depth):
            x = self.down_blocks[d](x)
            skips.append(x)
            x = self.downsample[d](x)

        x = self.bottom_block(x)
        
        for d in reversed(range(self.depth)):
            x = self.conv_trans[d](x)
            x = torch.cat((x, skips[d]), dim = 1)
            x = self.up_blocks[d](x)
            # results.append(self.output_convs[d](x))
        
        return self.output_conv(x)

if __name__ == '__main__':
    model = UNet2D()
    input = torch.rand(10,1,224,224)
    output = model(input)
    print(output.shape)