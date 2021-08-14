"""

"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb

import image_processing
import loss_model
import models
import output_handler
import plotting_helpers

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
G_WEIGHTS_FILE_NAME = 'netG.pth'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train(out_dir, real_img, scale_factor, total_scales, opt):
    """Train the entire scaled-model from a single image

    :param out_dir: the output base folder to create the scales folders in
    :param real_img: the real image to train from
    :param scale_factor: the actual (calculated) scaling factor between scales
    :param total_scales: the amount of scales to create
    :param opt: the configuration parameters for the network
    :return: tuple of (trained_generators, z_opts, noise_amps,  real_imgs)
                trained_generators - list of trained generators
                z_opts - list of padded optimal reconstruction noise(z)
                noise_amps - list of noise multipliers (amplitudes) with which each scale's
                             generator was trained
                real_imgs - list of the real image downscaled at each scale
    """

    real_imgs = image_processing.create_real_imgs_pyramid(real_img, scale_factor, total_scales,
                                                          opt)

    trained_generators = []
    z_opts = []
    noise_amps = []
    vgg = torchvision.models.vgg19(pretrained=True).features.to(opt.device).eval()
    nfc_prev = None

    for scale in range(total_scales):
        scale_out_dir = output_handler.gen_scale_dir(out_dir, scale)
        plotting_helpers.save_im(real_imgs[scale], scale_out_dir, 'real_scale', convert=True)

        curr_nfc = min(opt.nfc * pow(2, math.floor(scale / 4)), 128)
        curr_min_nfc = min(opt.min_nfc * pow(2, math.floor(scale / 4)), 128)

        curr_G = _init_generator(curr_nfc, curr_min_nfc, opt)

        # Learn initial weights guess from previous scale
        if nfc_prev == curr_nfc and opt.try_initial_guess:
            print("Initial weights guess is previous scale")
            prev_out_dir = output_handler.gen_scale_dir(out_dir, scale - 1)
            curr_G.load_state_dict(torch.load(os.path.join(prev_out_dir, G_WEIGHTS_FILE_NAME)))

        # Train a single scale
        start_time = time.time()
        curr_G, z_curr, curr_noise_amp = _train_single_scale(trained_generators, z_opts, noise_amps,
                                                             curr_G, real_imgs, vgg, scale_out_dir,
                                                             scale_factor, opt)
        print(f"{scale} Scale Training Time: {time.time()-start_time}")

        # Save the trained generator for future loading
        torch.save(curr_G.state_dict(), os.path.join(scale_out_dir, G_WEIGHTS_FILE_NAME))

        # Append results
        [p.requires_grad_(False) for p in curr_G.parameters()]
        curr_G.eval()
        trained_generators.append(curr_G)
        z_opts.append(z_curr)
        noise_amps.append(curr_noise_amp)

        # prepare next scale
        nfc_prev = curr_nfc

    return trained_generators, z_opts, noise_amps,  real_imgs


def _init_generator(curr_nfc, curr_min_nfc, opt):
    """create a single-scale generator and initialize it.

    :param curr_nfc: the current generator base number of output channels
    :param curr_min_nfc: the minimum amount of output channels allowed
    :param opt: the configuration parameters for the network
    :return: a single scale generator
    """

    net_g = models.GeneratorConcatSkip2CleanAdd(curr_nfc, opt.nc, opt.ker_size, opt.padd_size,
                                               opt.stride, opt.num_layer,
                                               curr_min_nfc).to(opt.device)
    net_g.apply(models.weights_init)
    # TODO load from file?
    return net_g


def _train_single_scale(trained_generators, z_opts, noise_amps, curr_g, real_imgs, vgg, out_dir,
                        scale_factor, opt):
    """Train a single-scale generator based on a single image

    :param trained_generators: list of trained generators so far
    :param z_opts: list of padded optimal reconstruction noise(z) so far
    :param noise_amps: list of noise multipliers (amplitudes) with which each scale's generator was
                       trained so far
    :param curr_g: the initialized single-scale generator to train
    :param real_imgs: list of the real image downscaled at each scale
    :param vgg: a pre-trained vgg network to extract features from
    :param out_dir: the output base folder for this scale
    :param scale_factor: the actual (calculated) scaling factor between scales
    :param opt: the configuration parameters for the network
    :return: tuple of (curr_g, z_opt, noise_amp)
                curr_g - the trained single-scale generator
                z_opt - padded optimal reconstruction noise(z)
                noise_amp - noise multiplier (amplitude) with which the scale generator was trained
    """

    cur_scale = len(trained_generators)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
          f"\t\tSCALE {cur_scale}\n"
          "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    real_img = real_imgs[cur_scale]
    opt.nzx = real_img.shape[2]  # Width of image in current scale
    opt.nzy = real_img.shape[3]  # Height of image in current scale

    # Create the padding layer
    pad_amount = models.get_pad_amount(opt.ker_size, opt.num_layer)
    pad_func = nn.ZeroPad2d(int(pad_amount))

    # Create and initialize the chosen loss block
    loss_block, layers_losses = loss_model.generate_loss_block(vgg, real_img, opt.loss_func,
                                                               opt.chosen_layers, opt)
    # Create a color loss block if needed
    c_loss_block = None
    if opt.c_alpha:
        c_loss_block = loss_model.generate_c_loss_block(real_img, opt.c_patch_size, opt.loss_func,
                                                        opt.nc, opt.device)

    # Setup Optimizer
    optimizer = optim.Adam(curr_g.parameters(),
                           lr=opt.lr * (opt.lr_factor ** cur_scale),
                           betas=(opt.beta1, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[int(0.8*opt.epochs)],
                                               gamma=opt.gamma)

    # Track losses for output graphs
    distribution_loss_arr = []
    rec_loss_arr = []
    color_loss_arr = []

    # z_opt is a padding of {Z*, 0, 0, 0, ...}. The specific set of input noise maps which
    # generates the original image xn
    if cur_scale:
        prev = _draw_concat(trained_generators, z_opts, real_imgs, noise_amps, 'rand', pad_func,
                            scale_factor, opt)
        prev = pad_func(prev)
        z_prev = _draw_concat(trained_generators, z_opts, real_imgs, noise_amps, 'rec', pad_func,
                              scale_factor, opt)
        criterion = nn.MSELoss()
        RMSE = torch.sqrt(criterion(real_img, z_prev))
        noise_amp = opt.noise_amp * RMSE
        z_prev = pad_func(z_prev)
        if opt.z_opt_zero:
            z_opt = pad_func(torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, dtype=torch.float32, device=opt.device))
        else:
            z_opt = image_processing.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
            z_opt = pad_func(z_opt.expand(1, opt.nc, opt.nzx, opt.nzy))
    else:
        prev = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
        # TODO in_s = prev
        prev = pad_func(prev)
        z_prev = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
        z_prev = pad_func(z_prev)
        noise_amp = 1
        z_opt = image_processing.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
        z_opt = pad_func(z_opt.expand(1, opt.nc, opt.nzx, opt.nzy))
        # Notice that the noise for the 3 RGB channels is the same

    if cur_scale == 0:
        example_noise = image_processing.generate_noise([1, opt.nzx, opt.nzy]).detach()
        example_noise = example_noise.expand(1, opt.nc, opt.nzx, opt.nzy)
    else:
        example_noise = image_processing.generate_noise([opt.nc, opt.nzx, opt.nzy]).detach()
    example_noise = pad_func(example_noise)

    start_time = time.time()
    style_rec_factor = 1
    for epoch in range(opt.epochs):
        # noise_ is the input noise (before adding the image or changing the variance)
        noise_ = image_processing.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
        noise_ = pad_func(noise_.expand(1, opt.nc, opt.nzx, opt.nzy))
        # Notice that the noise for the 3 RGB channels is the same

        noise = noise_*noise_amp + prev

        # TODO-THINK for every step in G steps
        loss = 0
        total_loss = 0
        rec_loss = 0
        for j in range(opt.Gsteps):
            curr_g.zero_grad()
            fake_im = curr_g(noise.detach(), prev)  # TODO think on detach

            if opt.upsample_for_vgg:
                fake_im = loss_model.validate_vgg_im_size(fake_im)
                n_layers = len(opt.chosen_layers)
            else:
                n_layers = len(loss_model.validate_vgg_layers_amount(
                    fake_im.shape[2:], opt.chosen_layers, opt.min_features))

            loss_block(fake_im)
            if opt.c_alpha != 0:
                fake_im_patches = loss_model.split_img_to_patches(fake_im, opt.c_patch_size)
                fake_im_patches_flattened = fake_im_patches.reshape(1, -1, opt.nc * opt.c_patch_size * opt.c_patch_size, 1)
                c_loss_block(fake_im_patches_flattened)
                color_loss = c_loss_block.loss
            else:
                color_loss = 0

            loss = color_loss * opt.c_alpha / (n_layers + 1)
            for i, sl in enumerate(layers_losses):
                loss += opt.layers_weights[i] * sl.loss / (n_layers + 1)
            distribution_loss_arr.append(loss.detach())
            #loss.backward(retain_graph=True)

            if opt.alpha != 0:
                Z_opt = noise_amp*z_opt + z_prev
                #               -->         z_opt = 0 ({Z*,0,0,0,0,0})
                loss_criterion = nn.MSELoss()
                #rec_loss = (5**cur_scale) * opt.alpha * loss_criterion(curr_G(Z_opt.detach(), z_prev), real_img)
                rec_loss = loss_criterion(curr_g(Z_opt.detach(), z_prev), real_img)
                # rec_loss.backward(retain_graph=True)
                # rec_loss = rec_loss.detach()
            else:
                Z_opt = noise_amp*z_opt
                rec_loss = 0

            total_loss = loss + opt.alpha*rec_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
        # scheduler.step(total_loss)
        rec_loss_arr.append(rec_loss.detach())
        color_loss_arr.append(color_loss.detach() if opt.c_alpha else color_loss)

        logging_dict = {f'scale_{cur_scale}': {'loss': distribution_loss_arr[-1], 'rec_loss': rec_loss_arr[-1], 'total_loss': total_loss.detach(), 'epoch': epoch},
                        'running_total_loss': total_loss.detach(), 'running_rec_loss': rec_loss_arr[-1], 'running_loss': distribution_loss_arr[-1]}

        if epoch % opt.epoch_print == 0:
            print_line = f"epoch {epoch}:\t{opt.loss_func}:%.2e \t Rec:%.2e \t Color:%.2e \t" \
                    "Time: %.2f" % (distribution_loss_arr[-1], rec_loss_arr[-1], color_loss_arr[-1],
                                    time.time() - start_time)
            print(print_line)
            with open(os.path.join(out_dir, 'log.txt'), 'a') as f:
                f.write(f'{print_line}\n')

            start_time = time.time()
        if opt.epoch_show != -1 and epoch % opt.epoch_show == 0:
            # ==== SOME BUGS IN COMMENTS (INPUT TO CURR_G SHOULD BE NOISEAMP*NOISE+PREV)  ====
            # ====                  UNCOMMENT CAREFULLY                                   ====
            # example_fake = curr_G(example_noise, prev)
            # plotting_helpers.show_im(example_fake, title=f'e{epoch} epoch')
            # details_fake = curr_G(example_noise, z_prev)
            # plotting_helpers.show_im(details_fake, title=f'Details {epoch} epoch')
            z_opt_fake = curr_g(Z_opt, z_prev)
            plotting_helpers.show_im(z_opt_fake, title=f'Zopt_e{epoch} epoch')
        if epoch % opt.epoch_save == 0:
            # example_fake = curr_G(example_noise, prev)
            # plotting_helpers.save_im(example_fake, out_dir, f'e{epoch}', convert=True)
            ex_noise = example_noise*noise_amp + z_prev
            details_fake = curr_g(ex_noise, z_prev)
            details_fake_wandb = wandb.Image(plotting_helpers.convert_im(details_fake), caption=f'Details_e{epoch}')
            # plotting_helpers.save_im(details_fake, out_dir, f'Details_e{epoch}', convert=True)
            z_opt_fake = curr_g(Z_opt, z_prev)
            z_opt_fake_wandb = wandb.Image(plotting_helpers.convert_im(z_opt_fake), caption=f'Zopt_e{epoch}')
            logging_dict[f'scale_{cur_scale}']['Z_opt']= z_opt_fake_wandb
            logging_dict[f'scale_{cur_scale}']['Details_fake'] = details_fake_wandb
            plotting_helpers.save_im(z_opt_fake, out_dir, f'Zopt_e{epoch}', convert=True)

        #wandb.log({'loss': loss, 'rec_loss': rec_loss, 'total_loss': total_loss, 'epoch': epoch,
        #           'scale': cur_scale})
        wandb.log(logging_dict)

        # update prev
        prev = _draw_concat(trained_generators, z_opts, real_imgs, noise_amps, 'rand', pad_func,
                            scale_factor, opt)
        prev = pad_func(prev)

    # ========================================== END OF ALL EPOCHS ================================================
    # TODO save network?
    fig = plotting_helpers.plot_losses(distribution_loss_arr, rec_loss_arr, show=(opt.epoch_show > -1))
    plotting_helpers.save_fig(fig, out_dir, 'fin')
    images_wandb = []
    images_wandb_all = []
    # ======== GENERATE THIS SCALE'S RANDOM SAMPLES =========
    for i in range(opt.generate_fake_amount):
        # Generate an image using the same "prev" image (i.e., only the last layer changes stuff)
        if cur_scale == 0:
            example_noise = image_processing.generate_noise([1, opt.nzx, opt.nzy]).detach()
            example_noise = example_noise.expand(1, opt.nc, opt.nzx, opt.nzy)
        else:
            example_noise = image_processing.generate_noise([opt.nc, opt.nzx, opt.nzy]).detach()
        example_noise = pad_func(example_noise)
        z_in = noise_amp * example_noise + prev
        example_fake = curr_g(z_in, prev)

        # Generate an image using a random "prev" image (i.e., the entire image should be different)
        example_prev = _draw_concat(trained_generators, z_opts, real_imgs, noise_amps, 'rand',
                                    pad_func, scale_factor, opt)
        example_prev = pad_func(example_prev)
        z_in = noise_amp * example_noise + example_prev
        example_fake_all = curr_g(z_in, example_prev)

        if opt.epoch_show != -1:
            fim = plotting_helpers.show_im(example_fake, title=f'Final Image - Same prev {i}')
            plotting_helpers.save_im(fim, out_dir, f'fake_samePrev{i}')
            faim = plotting_helpers.show_im(example_fake_all, title=f'Final Image{i}')
            plotting_helpers.save_im(faim, out_dir, f'fake_{i}')
        else:
            plotting_helpers.save_im(example_fake, out_dir, f'fake_samePrev{i}', convert=True)
            plotting_helpers.save_im(example_fake_all, out_dir, f'fake_{i}', convert=True)
        images_wandb.append(wandb.Image(plotting_helpers.convert_im(example_fake), caption=f'fake_samePrev{i}'))
        images_wandb_all.append(wandb.Image(plotting_helpers.convert_im(example_fake_all), caption=f'fake_{i}'))
    wandb.log({f'example_fake_{cur_scale}': images_wandb, f'example_fake_all_{cur_scale}': images_wandb_all})

    # details_fake = curr_G(example_noise, z_prev)
    z_opt_fake = curr_g(Z_opt, z_prev)
    if opt.epoch_show != -1:
        # fim = plotting_helpers.show_im(example_fake, title='Final Image')
        # dim = plotting_helpers.show_im(details_fake, title='Final Details Image')
        zim = plotting_helpers.show_im(z_opt_fake, title='Final Zopt Image')
        # plotting_helpers.save_im(fim, out_dir, 'fin')
        # plotting_helpers.save_im(dim, out_dir, 'details_fin')
        plotting_helpers.save_im(zim, out_dir, 'zopt_fin')
    else:
        # plotting_helpers.save_im(example_fake, out_dir, 'fin', convert=True)
        # plotting_helpers.save_im(details_fake, out_dir, 'details_fin', convert=True)
        plotting_helpers.save_im(z_opt_fake, out_dir, 'zopt_fin', convert=True)

    return curr_g, z_opt, noise_amp


def _draw_concat(trained_generators, z_opts, real_imgs, noise_amps, mode, pad_func,
                 scale_factor, opt):
    fake = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
    if len(trained_generators):
        if mode == 'rand':
            pad_amount = models.get_pad_amount(opt.ker_size, opt.num_layer)

            for i, (gen, z_opt, cur_real_im, next_real_im, noise_amp) in enumerate(zip(
                    trained_generators, z_opts, real_imgs, real_imgs[1:], noise_amps)):
                if i:
                    z = image_processing.generate_noise([opt.nc, z_opt.shape[2] - 2 * pad_amount,
                                                         z_opt.shape[3] - 2 * pad_amount],
                                                        device=opt.device)
                else:
                    z = image_processing.generate_noise([1, z_opt.shape[2] - 2 * pad_amount,
                                                         z_opt.shape[3] - 2 * pad_amount],
                                                        device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])

                z = pad_func(z)
                prev_fake = fake[:,:,:cur_real_im.shape[2], :cur_real_im.shape[3]]
                prev_fake = pad_func(prev_fake)
                z_in = noise_amp * z + prev_fake
                fake = gen(z_in.detach(), prev_fake)
                fake = image_processing.resize(fake, 1 / scale_factor, opt.nc, opt.is_cuda)
                fake = fake[:, :, :next_real_im.shape[2], :next_real_im.shape[3]]
        elif mode == 'rec':
            for gen, z_opt, cur_real_im, next_real_im, noise_amp in zip(trained_generators, z_opts,
                                                                        real_imgs, real_imgs[1:],
                                                                        noise_amps):
                prev_fake = fake[:,:,:cur_real_im.shape[2], :cur_real_im.shape[3]]  # Todo -check
                prev_fake = pad_func(prev_fake)
                z_in = noise_amp*z_opt + prev_fake
                fake = gen(z_in.detach(), prev_fake)
                fake = image_processing.resize(fake, 1/scale_factor, opt.nc, opt.is_cuda)
                fake = fake[:,:,:next_real_im.shape[2], :next_real_im.shape[3]]
    return fake
