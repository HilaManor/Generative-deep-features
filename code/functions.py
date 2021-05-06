# Author: Tamar Rott-Shaham, et al. SinGAN 2019
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import torch
from torch import nn
import numpy as np
from skimage import io as img
from skimage import color
import imresize
import math

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise


def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)


def read_image(image_path, im_channels_num, is_cuda):

    x = img.imread(image_path)
    x = np2torch(x, im_channels_num, is_cuda)
    x = x[:, 0:3, :, :]  # TODO WHY?
    return x


def np2torch(x, im_channels_num, is_cuda):
    if im_channels_num == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if is_cuda:
        x = x.to(torch.device('cuda')) # todo do we need to reassign
    x = x.type(torch.cuda.FloatTensor) if is_cuda else x.type(torch.FloatTensor)
    x = norm(x)
    return x


def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def resize(im, scale, nc, is_cuda):
    im = torch2uint8(im)
    im = imresize.imresize_in(im, scale_factor=scale)
    im = np2torch(im, nc, is_cuda)
    return im

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = inp[-1,:,:,:].to(torch.device('cpu'))
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = inp[-1,-1,:,:].to(torch.device('cpu'))
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp


def preprocess_image(real_img, opt):
    n_scales_curr2min = math.ceil(math.log(opt.min_size /
                                           (min(real_img.shape[2], real_img.shape[3])),
                                           opt.scale_factor)) + 1
    n_scales_curr2max = math.ceil(math.log(min([opt.max_size, max([real_img.shape[2], real_img.shape[3]])]) /
                                           max([real_img.shape[2], real_img.shape[3]]), opt.scale_factor))
    total_downsampling_scales = n_scales_curr2min - n_scales_curr2max
    initial_resize_scale = min(opt.max_size / max([real_img.shape[2], real_img.shape[3]]), 1)

    real_resized = resize(real_img, initial_resize_scale, opt.nc, opt.is_cuda)

    scale_factor = math.pow(opt.min_size/(min(real_resized.shape[2], real_resized.shape[3])),
                            1 / total_downsampling_scales)
    total_scales = total_downsampling_scales + 1
    return real_resized, scale_factor, total_scales

def create_real_imgs_pyramid(real_img, scale_factor, total_scales, opt):
    pyramid = []
    # TODO: real = real[:,0:3,:,:]
    for scale_power in reversed(range(total_scales)):
        scale = math.pow(scale_factor, scale_power)
        resized_real = resize(real_img, scale, opt.nc, opt.is_cuda)
        pyramid.append(resized_real)
    return pyramid
