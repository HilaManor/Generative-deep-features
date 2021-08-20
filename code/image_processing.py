# Author: Tamar Rott-Shaham, et al. SinGAN 2019
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import torch
import math
import image_helpers
import imresize


def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = image_helpers.upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise


def resize(im, scale, nc, is_cuda):
    im = image_helpers.torch2uint8(im)
    im = imresize.imresize_in(im, scale_factor=scale)
    im = image_helpers.np2torch(im, nc, is_cuda)
    return im


def resize_to_shape(im, output_shape, nc, is_cuda):
    im = image_helpers.torch2uint8(im)
    im = imresize.imresize_in(im, output_shape=output_shape)
    im = image_helpers.np2torch(im, nc, is_cuda)
    return im

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
