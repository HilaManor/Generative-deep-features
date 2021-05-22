import models
import loss_model
import torchvision
import torch

import math
import time
import torch.nn as nn
import torch.optim as optim
import plotting_helpers
import image_processing
import output_handler


def train(out_dir, real_img, scale_factor, total_scales, opt):
    real_imgs = image_processing.create_real_imgs_pyramid(real_img, scale_factor, total_scales, opt)

    trained_generators = []
    Zs = []
    noise_amps = []
    vgg = torchvision.models.vgg19(pretrained=True).features.to(opt.device).eval()
    for scale in range(total_scales):
        curr_nfc = min(opt.nfc * pow(2, math.floor(scale / 4)), 128)
        curr_min_nfc = min(opt.min_nfc * pow(2, math.floor(scale / 4)), 128)

        scale_out_dir = output_handler.gen_scale_dir(out_dir, scale)

        plotting_helpers.save_im(real_imgs[scale], scale_out_dir, f"real_scale.png", convert=True)

        curr_G = init_generator(curr_nfc, curr_min_nfc, opt)

        # TODO load state dict ??

        start_time = time.time()
        curr_G, z_curr, curr_noise_amp = train_single_scale(trained_generators, Zs, noise_amps,
                                                            Curr_G, real_imgs, vgg, scale_out_dir,
                                                            scale_factor, opt)
        print(f"{scale} Scale Training Time: {time.time()-start_time}")

        [p.requires_grad_(False) for p in curr_G.parameters()]
        curr_G.eval()
        trained_generators.append(curr_G)
        Zs.append(z_curr)
        noise_amps.append(curr_noise_amp)

        # TODO save trained
        # TODO -check del curr_G?

    return trained_generators, Zs


def init_generator(curr_nfc, curr_min_nfc, opt):
    netG = models.GeneratorConcatSkip2CleanAdd(curr_nfc, opt.nc, opt.ker_size, opt.padd_size,
                                               opt.stride, opt.num_layer,
                                               curr_min_nfc).to(opt.device)
    netG.apply(models.weights_init)
    # TODO load from file?
    print(netG)
    return netG


def train_single_scale(trained_generators, Zs, noise_amps, curr_G, real_imgs, vgg, out_dir, scale_factor, opt):
    real_img = real_imgs[len(trained_generators)]
    opt.nzx = real_img.shape[2]  # Width of image in current scale
    opt.nzy = real_img.shape[3]  # Height of image in current scale
    # TODO-FUTURE receptive field...
    # the padding amount is determined by the generators amount of layers
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

    noise_pad_func = nn.ZeroPad2d(int(pad_noise))
    image_pad_func = nn.ZeroPad2d(int(pad_image))

    loss_block, layers_losses = loss_model.generate_loss_block(vgg, real_img, opt.loss_func, opt.chosen_layers, opt)

    # Setup Optimizer
    optimizer = optim.Adam(curr_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600,1500,2600],
                                               gamma=opt.gamma)

    style_loss_arr = []
    rec_loss_arr = []

    # z_opt is {Z*, 0, 0, 0, ...}. The specific set of input noise maps
    # which generates the original image xn
    if len(trained_generators):
        prev = draw_concat(trained_generators, Zs, real_imgs, noise_amps, 'rand', noise_pad_func,
                           image_pad_func, scale_factor, opt)
        prev = image_pad_func(prev)
        z_prev = draw_concat(trained_generators, Zs, real_imgs, noise_amps, 'rec', noise_pad_func,
                             image_pad_func, scale_factor, opt)
        criterion = nn.MSELoss()
        RMSE = torch.sqrt(criterion(real_img, z_prev))
        noise_amp = opt.noise_amp * RMSE
        z_prev = image_pad_func(z_prev)
        z_opt = noise_pad_func(torch.full([opt.nc, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device))
    else:
        prev = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
        # TODO in_s = prev
        prev = image_pad_func(prev)
        z_prev = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
        z_prev = noise_pad_func(z_prev)
        noise_amp = 1
        z_opt = image_processing.generate_noise([1, opt.nzx, opt.nzy], device=opt.is_cuda)
        z_opt = noise_pad_func(z_opt.expand(1, opt.nc, opt.nzx, opt.nzy))
        # Notice that the noise for the 3 RGB channels is the same

    example_noise = image_processing.generate_noise([1, opt.nzx, opt.nzy]).detach()
    example_noise = noise_pad_func(example_noise.expand(1, opt.nc, opt.nzx, opt.nzy))

    start_time = time.time()
    for epoch in range(opt.epochs):
        # noise_ is the input noise (before adding the image or changing the variance)
        noise_ = image_processing.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
        noise_ = noise_pad_func(noise_.expand(1, opt.nc, opt.nzx, opt.nzy))
        # Notice that the noise for the 3 RGB channels is the same

        noise = noise_*noise_amp + prev

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

            if opt.alpha != 0:
                Z_opt = noise_amp*z_opt + z_prev
                #               -->         z_opt = 0 ({Z*,0,0,0,0,0})
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
        prev = draw_concat(trained_generators, Zs, real_imgs, noise_amps, 'rand', noise_pad_func,
                           image_pad_func, scale_factor, opt)
        prev = image_pad_func(prev)

    # TODO save network?
    fig = plotting_helpers.plot_losses(style_loss_arr, rec_loss_arr)
    plotting_helpers.save_fig(fig, out_dir, 'fin')
    example_fake = curr_G(example_noise, prev)
    im = plotting_helpers.show_im(example_fake, title='Final Image')
    plotting_helpers.save_im(im, out_dir, 'fin')
    im = plotting_helpers.show_im(example_fake, title='Final Image')
    plotting_helpers.save_im(im, out_dir, 'fin')

    return curr_G, z_opt, noise_amp


def draw_concat(trained_generators, Zs, real_imgs, noise_amps, mode, noise_pad_func,
                image_pad_func, scale_factor, opt):
    fake = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
    if len(trained_generators):
        if mode == 'rand':
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            z = image_processing.generate_noise([1, Zs[0].shape[2] - 2*pad_noise,
                                                 Zs[0].shape[3] - 2*pad_noise], device=opt.device)
            z = z.exapnd(1, 3, z.shape[2], z.shape[3])
            for gen, Z_opt, cur_real_im, next_real_im, noise_amp in zip(trained_generators, Zs,
                                                                        real_imgs, real_imgs[1:],
                                                                        noise_amps):
                z = noise_pad_func(z)
                prev_fake = fake[:,:,:cur_real_im.shape[2], :cur_real_im.shape[3]]
                prev_fake = image_pad_func(prev_fake)
                z_in = noise_amp * z + prev_fake
                fake = gen(z_in.detach(), prev_fake)
                fake = image_processing.resize(fake, 1 / scale_factor, opt.nc, opt.is_cuda)
                fake = fake[:, :, :next_real_im.shape[2], :next_real_im.shape[3]]
                z = image_processing.generate_noise([opt.nc, Z_opt.shape[2] - 2*pad_noise,
                                                 Z_opt.shape[3] - 2*pad_noise], device=opt.device)
        elif mode == 'rec':
            for gen, Z_opt, cur_real_im, next_real_im, noise_amp in zip(trained_generators, Zs,
                                                                        real_imgs, real_imgs[1:],
                                                                        noise_amps):
                prev_fake = fake[:,:,:cur_real_im.shape[2], :cur_real_im.shape[3]]  # Todo -check
                prev_fake = image_pad_func(prev_fake)
                z_in = noise_amp*Z_opt + prev_fake
                fake = gen(z_in.detach(), prev_fake)
                fake = image_processing.resize(fake, 1/scale_factor, opt.nc, opt.is_cuda)
                fake = fake[:,:,:next_real_im.shape[2], :next_real_im.shape[3]]
    return fake
