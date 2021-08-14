import torch
import torch.nn as nn
import numpy as np
import math
import gram_loss
import contextual_loss
import pd_loss

vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225])

STYLE_LAYERS_TRANSLATION = {'conv_1': 'conv1_1',
                            'conv_2': 'conv1_2',

                            'conv_3': 'conv2_1',
                            'conv_4': 'conv2_2',

                            'conv_5': 'conv3_1',
                            'conv_6': 'conv3_2',
                            'conv_7': 'conv3_3',
                            'conv_8': 'conv3_4',

                            'conv_9': 'conv4_1',
                            'conv_10': 'conv4_2',
                            'conv_11': 'conv4_3',
                            'conv_12': 'conv4_4',

                            'conv_13': 'conv5_1',
                            'conv_14': 'conv5_2',
                            'conv_15': 'conv5_3',
                            'conv_16': 'conv5_4'}


class Normalization(nn.Module):
    """create a module to normalize input image so we can easily put it in a nn.Sequential"""
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        """normalize img"""
        return (img - self.mean) / self.std


def generate_loss_block(vgg, real_img, mode, chosen_layers, opt):
    # TODO - check: vgg = copy.deepcopy(vgg)

    if opt.upsample_for_vgg:
        real_img = validate_vgg_im_size(real_img)
    else:
        chosen_layers = validate_vgg_layers_amount(real_img.shape[2:], chosen_layers, opt.min_features)

    # normalization module
    normalization = Normalization(vgg_normalization_mean.to(opt.device),
                                  vgg_normalization_std.to(opt.device)).to(opt.device)

    # an iterable access to or list of content/syle losses
    layers_losses = []

    # assuming that vgg is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # Attach losses to the chosen layers
        if name.startswith('conv') and STYLE_LAYERS_TRANSLATION[name] in chosen_layers:
            target_feature = model(real_img).detach()
            if mode.lower() == 'style':
                loss_f = gram_loss.GramLoss
            elif mode.lower() == 'contextual':
                loss_f = contextual_loss.ContextualLoss
            elif mode.lower() == 'pdl':
                loss_f = pd_loss.PDLoss
            else:
                raise RuntimeError(f'Unrecognized loss: {mode}')

            loss = loss_f(target_feature, device=opt.device)
            model.add_module(f"{mode.lower()}_loss_{i}", loss)
            layers_losses.append(loss)  # Appends by reference :)

    # now we trim off the layers after the last losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], loss_f):
            break

    model = model[:(i + 1)]
    return model, layers_losses


def validate_vgg_im_size(im):
    min_d = min(list(im.shape[2:]))
    if min_d < 224:
        scale_factor = 224 / min_d
        im = nn.functional.interpolate(im, scale_factor=(scale_factor, scale_factor), mode='bilinear', align_corners=False, recompute_scale_factor=False)
    return im


def validate_vgg_layers_amount(im_shape, layers, min_features):
    n = math.floor(1 + math.log(im_shape[0]*im_shape[1] / min_features))
    return layers[0:n]  # if n > len(layers), returns all layers.


def generate_c_loss_block(real_img, c_patch_size, mode, nc, device):
    real_img_patches = split_img_to_patches(real_img, c_patch_size)
    real_img_patches_flattened = real_img_patches.reshape(1, -1, nc * c_patch_size * c_patch_size, 1)

    if mode.lower() == 'gram':
        loss_f = gram_loss.GramLoss
    elif mode.lower() == 'contextual':
        loss_f = contextual_loss.ContextualLoss
    elif mode.lower() == 'pdl':
        loss_f = pd_loss.PDLoss

    return loss_f(real_img_patches_flattened, device=device)


def split_img_to_patches(im, patch_size):
    patches = im.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute((0, 2, 3, 1, 4, 5))
    patches = patches.reshape(-1, 3, patch_size, patch_size)
    return patches
