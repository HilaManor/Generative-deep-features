import models
import loss_model
import functions
import torchvision
import torch

import time
import torch.nn as nn
import torch.optim as optim
import plotting_helpers
import functions

def train(out_dir, real_img, scale_factor, total_scales, opt):
    real_imgs = functions.create_real_imgs_pyramid(real_img, scale_factor, total_scales, opt)

    Generators = []
    Zs = []

    for scale in range(total_scales):

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
    netG = models.GeneratorConcatSkip2CleanAdd(opt.nfc, opt.nc, opt.ker_size, opt.padd_size, opt.stride, opt.num_layer, opt.min_nfc).to(opt.device)
    #netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
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
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600,1500,2600],
                                               gamma=opt.gamma)

    # TODO arrays for errors for graphs
    style_loss_arr = []
    rec_loss_arr = []

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

    # z_opt is {Z*, 0, 0, 0, ...}. The specific set of input noise maps
    # which generates the original image xn
    z_opt = functions.generate_noise([1, opt.nzx, opt.nzy])
    z_opt = noise_pad_func(z_opt.expand(1, opt.nc, opt.nzx, opt.nzy))
    # Notice that the noise for the 3 RGB channels is the same

    example_noise = functions.generate_noise([1, opt.nzx, opt.nzy]).detach()
    example_noise = noise_pad_func(example_noise.expand(1, opt.nc, opt.nzx, opt.nzy))

    start_time = time.time()
    for epoch in range(opt.epochs):
        # noise_ is the input noise (before adding the image or changing the variance)
        noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
        noise_ = noise_pad_func(noise_.expand(1, opt.nc, opt.nzx, opt.nzy))
        # Notice that the noise for the 3 RGB channels is the same

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
            style_loss_arr.append(loss.detach())
            loss.backward(retain_graph=True)

            # TODO-FUTUE implement reconstruction-loss using alpha (We want to ensure that there exists a specific set of input
            #  noise maps, which generates the original image x)

            if opt.alpha != 0:
                Z_opt = z_opt + z_prev
                #               -->         z_opt = 0 ({Z*,0,0,0,0,0})
                # Todo-future:    Z_opt = noise_amp * z_opt + z_prev
                loss_criterion = nn.MSELoss()
                rec_loss = opt.alpha * loss_criterion(curr_G(Z_opt.detach(), z_prev), real_img)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizer.step()
        scheduler.step()
        rec_loss_arr.append(rec_loss)

        if epoch % opt.epoch_print == 0:
            print(f"epoch {epoch}:\t{opt.loss_func}:%.2f \t Rec:%.2f \tTime: %.2f" %
                  (style_loss_arr[-1], rec_loss_arr[-1], time.time() - start_time))
            start_time = time.time()
        if epoch % opt.epoch_show == 0:
            example_fake = curr_G(example_noise, prev)
            plotting_helpers.show_im(example_fake, title=f'e{epoch} epoch')
            z_opt_fake = curr_G(z_opt, prev)
            plotting_helpers.show_im(z_opt_fake, title=f'Zopt_e{epoch} epoch')
        if epoch % opt.epoch_save == 0:
            example_fake = curr_G(example_noise, prev)
            plotting_helpers.save_im(example_fake, out_dir, f'e{epoch}', convert=True)
            z_opt_fake = curr_G(z_opt, prev)
            plotting_helpers.save_im(z_opt_fake, out_dir, f'Zopt_fin', convert=True)

        # update prev
        prev = draw_concat(Generators, 'rand', noise_pad_func, image_pad_func, opt)
        prev = image_pad_func(prev)

    # TODO save network?
    fig = plotting_helpers.plot_losses(style_loss_arr, rec_loss_arr)
    plotting_helpers.save_fig(fig, out_dir, 'fin')
    example_fake = curr_G(example_noise, prev)
    im = plotting_helpers.show_im(example_fake, title='Final Image')
    plotting_helpers.save_im(im, out_dir, 'fin')
    im = plotting_helpers.show_im(example_fake, title='Final Image')
    plotting_helpers.save_im(im, out_dir, 'fin')

    return curr_G, z_opt


def draw_concat(Generators, mode, noise_pad_func, image_pad_func, opt):
    # TODO-FUTURE good luck
    return torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
