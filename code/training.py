import models
import loss_model
import functions
import torchvision
import torch
import os

import time
import torch.nn as nn
import torch.optim as optim
import plotting_helpers

def train(real_img, out_dir, opt):
    real_imgs = [real_img]  # TODO-FUTURE created multi-scale images

    Generators = []
    Zs = []

    # TODO-FUTURE Create SCALES (while)
    #TODO (there's a bug already) plt.imsave(os.path.join(out_dir, "real_scale.png"))

    curr_G = init_generator(opt)
    vgg = torchvision.models.vgg19(pretrained=True).features.to(opt.device).eval()

    start_time = time.time()
    ## TODO-FUTURE - use diffrent dir for each scale
    curr_G, z_curr = train_single_scale(Generators, curr_G, real_imgs, vgg, out_dir, opt)
    print(f"{len(Generators)} Scale Training Time: {time.time()-start_time}")

    [p.requires_grad_(False) for p in curr_G.parameters()]
    curr_G.eval()
    Generators.append(curr_G)
    Zs.append(z_curr)

    return Generators, Zs


def init_generator(opt):
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    # TODO load from file?
    print(netG)
    return netG



def train_single_scale(Generators, curr_G, real_imgs, vgg, out_dir, opt):
    real_img = real_imgs[len(Generators)]
    opt.nzx = real_img.shape[2]  # Width of image in current scale
    opt.nzy = real_img.shape[3]  # Height of image in current scale
    # TODO-FUTURE receptive field...
    # the padding amount is determined by the generators amount of layers
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

    noise_pad_func = nn.ZeroPad2d(int(pad_noise))
    image_pad_func = nn.ZeroPad2d(int(pad_image))

    # TODO-FUTURE create noise (and pad it?)

    loss_block, layers_losses = loss_model.generate_loss_block(vgg, real_img, opt.loss_func, opt.chosen_layers, opt)

    # Setup Optimizer
    optimizer = optim.Adam(curr_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600,1800,3600],
                                               gamma=opt.gamma)

    # TODO arrays for errors for graphs
    style_loss = []

    # TODO-FUTURE currently only for generic implementation (1 scale)
    prev = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
    # TODO in_s = prev
    prev = image_pad_func(prev)
    z_prev = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
    z_prev = noise_pad_func(z_prev)
    # TODO-FUTURE opt.noise_amp = 1

    # TODO-FUTRE:
    #   first epoch & first step (D)
    #       first scale:
    #           in_s = zeros
    #           prev = padded zeros (by image)
    #           z_prev = padded zeros (by noise)
    #           noise_amp = 1
    #       else:
    #           prev = padded draw_concat 'rand' (by image)
    #               --> prev = previous generated image FROM RANDOM NOISE
    #           z_prev = padded draw_concat 'rec' (by image)
    #               --> z_prev = previous generate image FROM KNOWN NOISE
    #   else (not first epoch || not first step)

    start_time = time.time()
    for epoch in range(opt.epochs):
        # z_opt is {Z*, 0, 0, 0, ...}. The specific set of input noise maps
        # which generates the original image xn
        z_opt = functions.generate_noise([1, opt.nzx, opt.nzy])
        z_opt = noise_pad_func(z_opt.expand(1, opt.nc, opt.nzx, opt.nzy))

        # noise_ is the input noise (before adding the image or changing the variance)
        noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
        noise_ = noise_pad_func(noise_.expand(1, opt.nc, opt.nzx, opt.nzy))
        # Notice that the noise for the 3 RGB channels is the same
        # TODO-FUTURE z_opt should only be generated in 1st scale

        noise = noise_
        # TODO-FUTURE if not first scale noise = noise_amp * noise_ + prev


        # TODO-THINK for every step in G steps
        for j in range(opt.Gsteps):
            curr_G.zero_grad()
            fake_im = curr_G(noise.detach(), prev)  # TODO think on detach

            loss_block(fake_im)
            loss = 0
            for i, sl in enumerate(layers_losses):
                loss += opt.layers_weights[i] * sl.loss
            style_loss.append(loss)
            loss.backward()  # TODO retain_graph=True

            # TODO-FUTUE implement reconstruction-loss using alpha (We want to ensure that there exists a specific set of input
            #  noise maps, which generates the original image x)
            #      if alpha != 0
            #           Z_opt = noise_amp * z_opt + z_prev
            #               --> in first scale: z_prev=0, z_opt random every iter
            #               --> else: z_prev = previous generate image FROM KNOWN NOISE
            #               -->         z_opt = 0 ({Z*,0,0,0,0,0})
            #           GDS (train) by reconstruction loss
            #               --> .
            #       else:
            #           Z_opt = z_opt (remember, only in first scale the noise changes every ITER)
            #               --> = 0 in all but first scale
            #           rec_loss = 0

            optimizer.step()
        scheduler.step()

        if epoch % opt.epoch_print == 0:
            print("epoch {}:\t{}: {:4f}".format(epoch, opt.loss_func, style_loss[-1]), end="\t\t")
            print(f"Time: {time.time() - start_time}")
            start_time = time.time()

        # update prev
        prev = draw_concat(Generators, 'rand', noise_pad_func, image_pad_func, opt)
        prev = image_pad_func(prev)

    # TODO save network?
    fig = plotting_helpers.plot_loss(style_loss)
    plotting_helpers.save_fig(fig, out_dir, opt)
    return curr_G, z_opt


def draw_concat(Generators, mode, noise_pad_func, image_pad_func, opt):
    # TODO-FUTURE good luck
    return torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
