"""Mainly the code from SinGAN, so we respected the documentation. Only adding where we changed
SinGAN's code: https://github.com/tamarott/SinGAN"""

# Author: Tamar Rott-Shaham, et al. SinGAN 2019
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import torch
import torch.nn as nn
#import numpy as np
#import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv',
                        nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride,
                                  padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, nfc, nc, ker_size, padd_size, stride, num_layers, min_nfc):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        N = nfc
        self.head = ConvBlock(nc, N, ker_size, padd_size, stride)
        self.body = nn.Sequential()
        for i in range(num_layers - 2):
            N = int(nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, min_nfc), max(N, min_nfc), ker_size,
                              padd_size, stride)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, min_nfc), nc, kernel_size=ker_size, stride=stride,
                      padding=padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y


def get_pad_amount(ker_size, num_layer, pad_type):
    """Decide the padding amount of the image before being inserted to the generator
    
    :param ker_size: the kernel size of the convolutional layers in the generators
    :param num_layer: the amount of convolutional layers in the generator
    :param pad_type: the padding type chosen by the user. if 'between' was chosen, 
                     no need to padd in advance. if 'pre-padding' was given, use 
                     the calculated padding
    :return: the amount of padding to add to the image before inserting to the generator
    """
    if pad_type == 'pre-padding':
        return int(((ker_size - 1) * num_layer) / 2)
    elif pad_type == 'between':
        return 0
    else:
        raise NotImplementedError
