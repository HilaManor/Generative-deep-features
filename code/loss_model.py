import torch
import torch.nn as nn
import style_loss
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
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        """normalize img"""
        return (img - self.mean) / self.std


def generate_loss_block(vgg, real_img, mode, chosen_layers, opt):
    # TODO: vgg = copy.deepcopy(vgg)

    # normalization module
    normalization = Normalization(opt.normalization_mean.to(opt.device),
                                  opt.normalization_std.to(opt.device)).to(opt.device)

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
                loss_f = style_loss.StyleLoss
            elif mode.lower() == 'contextual':
                loss_f = contextual_loss.ContextualLoss
            elif mode.lower() == 'pdl':
                loss_f = pd_loss.PDLoss
            else:
                raise RuntimeError(f'Unrecognized loss: {mode}')

            loss = loss_f(target_feature, device=opt.device)
            model.add_module(f"{mode.lower()}_loss_{i}", loss)
            layers_losses.append(loss)

    # now we trim off the layers after the last losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], loss_f):
            break

    model = model[:(i + 1)]
    return model, layers_losses
