import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def init_weights(net, init_type='normal', gain=0.02):
    from torch.nn import init

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True, norm='batch_norm',
                 activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2d, self).__init__()

        self.norm = norm

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.mask_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups,  bias=bias)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.activation = activation

    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input_data):
        x = self.conv2d(input_data)
        mask = self.mask_conv2d(input_data)

        if self.norm == 'batch_norm':
            x = self.batch_norm2d(x)

        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)

        return x


class DeGatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0, bias=True, batch_norm=True,
                 activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(DeGatedConv, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.mask_deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding,
                                              output_padding=output_padding, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.activation = activation

    def forward(self, input_data):
        x = self.deconv(input_data)
        mask = self.mask_deconv(input_data)

        x = self.activation(x) * self.sigmoid(mask)

        return x


class SNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(SNConv, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = torch.nn.LeakyReLU(0.2, inplace=True)

    def forward(self, data):
        x = self.conv2d(data)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x





