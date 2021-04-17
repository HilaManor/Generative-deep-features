import time

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import copy
import helpers

import style_loss as StlLoss
import contextual_loss as CtxLoss
import pd_loss as PDLoss
import config_params

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


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


def get_style_model_and_losses(device, cnn, normalization_mean, normalization_std,
                               style_img, #content_img,
                               #content_layers,
                               style_layers):
    cnn = copy.deepcopy(cnn)  # HM Why?

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle losses
#     content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
      

        model.add_module(name, layer)

#         if name in content_layers:
#             # add content loss:
#             target = model(content_img).detach()
#             content_loss = ContentLoss(target)
#             model.add_module("content_loss_{}".format(i), content_loss)
#             content_losses.append(content_loss)

        if name.startswith('conv') and STYLE_LAYERS_TRANSLATION[name] in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
#            style_loss = StlLoss.StyleLoss(target_feature)
#             model.add_module("style_loss_{}".format(i), style_loss)
#             contx_loss = CtxLoss.ContextualLoss(target_feature, device=device)
#             model.add_module("contx_loss_{}".format(i), contx_loss)
            pd_loss = PDLoss.PDLoss(target_feature, device=device)
            model.add_module("pd_loss_{}".format(i), pd_loss)

#             style_losses.append(style_loss)
#             style_losses.append(contx_loss)
            style_losses.append(pd_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
#         if isinstance(model[i], StlLoss.StyleLoss): #or isinstance(model[i], ContentLoss):
#         if isinstance(model[i], CtxLoss.ContextualLoss): #or isinstance(model[i], ContentLoss):
#             break
        if isinstance(model[i], PDLoss.PDLoss): #or isinstance(model[i], ContentLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses#, content_losses
#     return model, style_losses#, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    if config_params.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam([input_img.requires_grad_()], lr=config_params.lr)
    else:
        optimizer = optim.LBFGS([input_img.requires_grad_()], lr=config_params.lr)
    return optimizer


## ~~~~~~~~~~~~~~ Main Training Loop ~~~~~~~~~~~~~
def run_style_transfer(device, cnn, normalization_mean, normalization_std,
                       #content_img, 
                       style_img, input_img,
                       style_layers, style_weights,
                       num_steps=300, style_weight=1000000):#, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
#     model, style_losses, content_losses = get_style_model_and_losses(cnn,
    model, style_losses = get_style_model_and_losses(device, cnn, 
        normalization_mean, normalization_std, style_img, style_layers)#, content_img)
    optimizer = get_input_optimizer(input_img)
    
    print('Optimizing..')
    run = [0]
    losses = []
    start_time = [time.time()]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
#             content_score = 0

            for i, sl in enumerate(style_losses):
                style_score += style_weights[i] * sl.loss
#             for cl in content_losses:
#                 content_score += cl.loss

            style_score *= style_weight
#             content_score *= content_weight

            loss = style_score # + content_score
            losses.append(float(loss))
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:\t\tStyle loss: {:4f}".format(run, style_score.item()),end="\t\t")
                print(f"Time: {time.time()-start_time[0]}")
                start_time[0] = time.time()
#                 print('Style Loss : {:4f} Content Loss: {:4f}'.format(
#                     style_score.item(), content_score.item()))
                #print('Style Loss : {:4f} '.format(style_score.item()))
                #print()
            if run[0] % config_params.imshow_cycles == 0:
                vis_img = copy.deepcopy(input_img)
                #plt.figure()
                helpers.imshow(vis_img, title=f'On run {run[0]}')
        
            return style_score #+ content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img, losses
