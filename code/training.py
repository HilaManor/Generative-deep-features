"""Create and training functions for the model.
function train - train the entire model
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
import numpy as np

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

        curr_g = _init_generator(curr_nfc, curr_min_nfc, opt)

        # Learn initial weights guess from previous scale
        if nfc_prev == curr_nfc and opt.try_initial_guess:
            print("Initial weights guess is previous scale")
            prev_out_dir = output_handler.gen_scale_dir(out_dir, scale - 1)
            curr_g.load_state_dict(torch.load(os.path.join(prev_out_dir, G_WEIGHTS_FILE_NAME)))

        # Train a single scale
        start_time = time.time()
        curr_g, z_curr, curr_noise_amp = _train_single_scale(trained_generators, z_opts,
                                                             noise_amps, curr_g, real_imgs, vgg,
                                                             scale_out_dir, scale_factor, opt)
        print(f"{scale} Scale Training Time: {time.time()-start_time}")

        # Save the trained generator for future loading
        torch.save(curr_g.state_dict(), os.path.join(scale_out_dir, G_WEIGHTS_FILE_NAME))

        # Append results
        [p.requires_grad_(False) for p in curr_g.parameters()]
        curr_g.eval()
        trained_generators.append(curr_g)
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
                        scale_factor, opt, centers=None):
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
    pad_amount = models.get_pad_amount(opt.ker_size, opt.num_layer, opt.pad_type)
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
        prev_rand = _draw_concat(trained_generators, z_opts, real_imgs, noise_amps, 'rand',
                                 pad_func, scale_factor, opt)
        prev_recon = _draw_concat(trained_generators, z_opts, real_imgs, noise_amps, 'rec',
                                  pad_func, scale_factor, opt)
        criterion = nn.MSELoss()
        rmse = torch.sqrt(criterion(real_img, prev_recon))
        noise_amp = opt.noise_amp * rmse
        if opt.z_opt_zero:
            z_opt = pad_func(torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, dtype=torch.float32,
                                        device=opt.device))
        else:
            z_opt = image_processing.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
            z_opt = pad_func(z_opt.expand(1, opt.nc, opt.nzx, opt.nzy))
            # Notice that the noise for the 3 RGB channels is the same
    else:
        prev_rand = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
        prev_recon = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
        noise_amp = 1
        z_opt = image_processing.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
        z_opt = pad_func(z_opt.expand(1, opt.nc, opt.nzx, opt.nzy))
        # Notice that the noise for the 3 RGB channels is the same
    prev_rand = pad_func(prev_rand)
    prev_recon = pad_func(prev_recon)


    # # new noise to generate details-examples from over the epochs
    # if not cur_scale:
    #     # In the first scale it should be constant across the channels
    #     example_noise = image_processing.generate_noise([1, opt.nzx, opt.nzy]).detach()
    #     example_noise = example_noise.expand(1, opt.nc, opt.nzx, opt.nzy)
    # else:
    #     example_noise = image_processing.generate_noise([opt.nc, opt.nzx, opt.nzy]).detach()
    # example_noise = pad_func(example_noise)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN TRAINING LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_time = time.time()
    for epoch in range(opt.epochs):
        if centers is not None:
            prev_rand = image_processing.quant2centers(prev_rand, centers, opt.device)
            plotting_helpers.save_im(prev_rand,out_dir, "prev_quant.png",convert=True)
        # noise_ is the input noise (before adding the image or changing the variance)
        if cur_scale > 0:
            noise_ = image_processing.generate_noise([opt.nc, opt.nzx, opt.nzy], device=opt.device)
        else:
            noise_ = image_processing.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
            noise_ = noise_.expand(1, opt.nc, opt.nzx, opt.nzy)
            # Notice that the noise for the 3 RGB channels is the same
        noise_ = pad_func(noise_)

        noise = noise_*noise_amp + prev_rand

        total_loss = 0
        for j in range(opt.Gsteps):
            curr_g.zero_grad()
            fake_im = curr_g(noise.detach(), prev_rand)  # Generate a fake image

            # Calculate loss
            if opt.c_alpha != 0:
                fake_im_patches = loss_model.split_img_to_patches(fake_im, opt.c_patch_size)
                fake_im_patches_flattened = fake_im_patches.reshape(1, -1, opt.nc *
                                                                    opt.c_patch_size *
                                                                    opt.c_patch_size, 1)
                c_loss_block(fake_im_patches_flattened)
                color_loss = c_loss_block.loss
            else:
                color_loss = 0

            if opt.upsample_for_vgg:
                fake_im = loss_model.validate_vgg_im_size(fake_im)
                n_layers = len(opt.chosen_layers)
            else:
                n_layers = len(loss_model.validate_vgg_layers_amount(
                    fake_im.shape[2:], opt.chosen_layers, opt.min_features))
            loss_block(fake_im)

            norm_const = opt.c_alpha + np.sum(opt.layers_weights[:n_layers])
            loss = color_loss * opt.c_alpha / norm_const
            color_loss_arr.append(float(loss.detach()) if opt.c_alpha else color_loss)
            for i, sl in enumerate(layers_losses):
                loss += opt.layers_weights[i] * sl.loss / norm_const
            distribution_loss_arr.append(float(loss.detach()))

            if opt.alpha != 0:
                if centers is not None:
                    prev_recon = image_processing.quant2centers(prev_recon, centers, opt.device)
                    plotting_helpers.save_im(prev_recon, out_dir,"z_prev.png",convert=True)
                Z_opt = noise_amp*z_opt + prev_recon
                #               -->         z_opt = 0 ({Z*,0,0,0,0,0})
                loss_criterion = nn.MSELoss()
                rec_loss = loss_criterion(curr_g(Z_opt.detach(), prev_recon), real_img)
                # rec_loss.backward(retain_graph=True)
                rec_loss_arr.append(rec_loss.detach())
            else:
                Z_opt = noise_amp*z_opt
                rec_loss = 0

            total_loss = loss + opt.alpha*rec_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LOGGING  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        logging_dict = {f'scale_{cur_scale}': {'loss': distribution_loss_arr[-1],
                                               'rec_loss': rec_loss_arr[-1],
                                               'total_loss': total_loss.detach(),
                                               'epoch': epoch},
                        'running_total_loss': total_loss.detach(),
                        'running_rec_loss': rec_loss_arr[-1],
                        'running_loss': distribution_loss_arr[-1]}

        # Print to stdout and log file
        if epoch % opt.epoch_print == 0:
            print_line = f"epoch {epoch}:\t{opt.loss_func}:%.2e \t Rec:%.2e \t Color:%.2e \t" \
                         "Time: %.2f" % (distribution_loss_arr[-1], rec_loss_arr[-1],
                                         color_loss_arr[-1], time.time() - start_time)
            print(print_line)
            with open(os.path.join(out_dir, 'log.txt'), 'a') as f:
                f.write(f'{print_line}\n')

            start_time = time.time()

        # Save each configured "save epoch" the current reconstruction
        if epoch % opt.epoch_save == 0:
            # ex_noise = example_noise*noise_amp + prev_recon
            # details_fake = curr_g(ex_noise, prev_recon)
            # details_fake_wandb = wandb.Image(plotting_helpers.convert_im(details_fake),
            #                                  caption=f'Details_e{epoch}')
            # plotting_helpers.save_im(details_fake, out_dir, f'Details_e{epoch}', convert=True)
            # logging_dict[f'scale_{cur_scale}']['Details_fake'] = details_fake_wandb
            z_opt_fake = curr_g(Z_opt, prev_recon)
            if centers is None:
                z_opt_fake_wandb = wandb.Image(plotting_helpers.convert_im(z_opt_fake),
                                               caption=f'Zopt_e{epoch}')
                logging_dict[f'scale_{cur_scale}']['Z_opt'] = z_opt_fake_wandb
            plotting_helpers.save_im(z_opt_fake, out_dir, f'Zopt_e{epoch}', convert=True)
        if centers is None:
            wandb.log(logging_dict)

        # update prev_rand
        prev_rand = _draw_concat(trained_generators, z_opts, real_imgs, noise_amps, 'rand',
                                 pad_func, scale_factor, opt)
        prev_rand = pad_func(prev_rand)

    # ===================================== END OF ALL EPOCHS =====================================
    fig = plotting_helpers.plot_losses(distribution_loss_arr, rec_loss_arr,
                                       show=(opt.epoch_show > -1))
    plotting_helpers.save_fig(fig, out_dir, 'fin')
    if centers is None:
        images_wandb = []
        images_wandb_all = []
    # ======== GENERATE THIS SCALE'S RANDOM SAMPLES =========
    for i in range(opt.generate_fake_amount):
        # Generate an image using the same "prev_rand" image
        #   (i.e., only the last layer changes stuff)
        if cur_scale > 0:
            example_noise = image_processing.generate_noise([opt.nc, opt.nzx, opt.nzy]).detach()
        else:
            example_noise = image_processing.generate_noise([1, opt.nzx, opt.nzy]).detach()
            example_noise = example_noise.expand(1, opt.nc, opt.nzx, opt.nzy)
        example_noise = pad_func(example_noise)
        z_in = noise_amp * example_noise + prev_rand
        example_fake = curr_g(z_in, prev_rand)

        # Generate an image using a random "prev_rand" image
        #   (i.e., the entire image should be different)
        example_prev = _draw_concat(trained_generators, z_opts, real_imgs, noise_amps, 'rand',
                                    pad_func, scale_factor, opt)
        example_prev = pad_func(example_prev)
        z_in = noise_amp * example_noise + example_prev
        example_fake_all = curr_g(z_in, example_prev)

        if opt.epoch_show != -1:
            fim = plotting_helpers.show_im(example_fake, title=f'Final Image - Same prev_rand {i}')
            plotting_helpers.save_im(fim, out_dir, f'fake_samePrev{i}')
            faim = plotting_helpers.show_im(example_fake_all, title=f'Final Image{i}')
            plotting_helpers.save_im(faim, out_dir, f'fake_{i}')
        else:
            plotting_helpers.save_im(example_fake, out_dir, f'fake_samePrev{i}', convert=True)
            plotting_helpers.save_im(example_fake_all, out_dir, f'fake_{i}', convert=True)
        if centers is None:
            images_wandb.append(wandb.Image(plotting_helpers.convert_im(example_fake),
                                            caption=f'fake_samePrev{i}'))
            images_wandb_all.append(wandb.Image(plotting_helpers.convert_im(example_fake_all),
                                                caption=f'fake_{i}'))
    if centers is None:
        wandb.log({f'example_fake_{cur_scale}': images_wandb,
                   f'example_fake_all_{cur_scale}': images_wandb_all})

    # Save the final reconstruction image
    z_opt_fake = curr_g(Z_opt, prev_recon)
    plotting_helpers.save_im(z_opt_fake, out_dir, 'zopt_fin', convert=True)

    return curr_g, z_opt, noise_amp


def _draw_concat(trained_generators, z_opts, real_imgs, noise_amps, mode, pad_func,
                 scale_factor, opt):
    """Generates an upscaled fake previous image using all the given scales generators
    For the first scale (no trained generators) will output a zeros image

    :param trained_generators: list of trained generators
    :param z_opts: list of padded optimal reconstruction noise(z)
    :param real_imgs: list of the real image downscaled at each scale
    :param noise_amps: list of noise multipliers (amplitudes) with which each scale's generator was
                       trained
    :param mode: 'rand' - create a completely new image by using random noise samples
                 'rec' - create a reconstruction image (using the z_opts)
    :param pad_func: the padding function for each image and noise image inputted to the generators
    :param scale_factor: the actual (calculated) scaling factor between scales
    :param opt: the configuration parameters for the network
    :return: a fake previous image generated given the mode (already upscaled)
    """

    fake = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
    if len(trained_generators):
        pad_amount = models.get_pad_amount(opt.ker_size, opt.num_layer, opt.pad_type)
        for i, (gen, z_opt, cur_real_im, next_real_im, noise_amp) in enumerate(zip(
                trained_generators, z_opts, real_imgs, real_imgs[1:], noise_amps)):
            prev_fake = fake[:, :, :cur_real_im.shape[2], :cur_real_im.shape[3]]
            prev_fake = pad_func(prev_fake)

            # The noise is either random or the optimal recon noise the generator was trained for
            if mode == 'rand':
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
            elif mode == 'rec':
                z = z_opt

            z_in = noise_amp * z + prev_fake
            fake = gen(z_in.detach(), prev_fake)
            fake = image_processing.resize(fake, 1 / scale_factor, opt.nc, opt.is_cuda)
            fake = fake[:, :, :next_real_im.shape[2], :next_real_im.shape[3]]
    return fake

def train_paint(opt,Generators,z_opts,reals,NoiseAmp,centers,paint_inject_scale, out_dir, total_scales, scale_factor):
    """Generates an upscaled fake previous image using all the given scales generators
    For the first scale (no trained generators) will output a zeros image
	
	:param opt: the configuration parameters for the network
    :param trained_generators: list of trained generators
    :param z_opts: list of padded optimal reconstruction noise(z)
    :param reals: list of the real image downscaled at each scale
    :param NoiseAmp: list of noise multipliers (amplitudes) with which each scale's generator was
                       trained
    :param centers: list of centers in the RGB space for quantizing the real image.
    :param paint_inject_scale: the scale in the generators pyramid from which the
							model is re-trained with quantized real image.
	:param out_dir: the output base folder for this scale
    :param total_scales: the amount of scales to create
    :param scale_factor: the actual (calculated) scaling factor between scales
    
    :return: retrained Generators, z_opt, NoiseAmp, reals for the quantized mode
    """
    vgg = torchvision.models.vgg19(pretrained=True).features.to(opt.device).eval()

    for scale_num in range(total_scales):
        curr_nfc = min(opt.nfc * pow(2, math.floor(scale_num / 4)), 128)
        curr_min_nfc = min(opt.min_nfc * pow(2, math.floor(scale_num / 4)), 128)
        if scale_num!=paint_inject_scale:
            continue

        scale_out_dir = output_handler.gen_scale_dir(out_dir, scale_num)

        plotting_helpers.save_im(reals[scale_num],scale_out_dir,"in_scale.png",convert=True)

        G_curr = _init_generator(curr_nfc, curr_min_nfc, opt)
        G_curr,z_opt,noise_amp = _train_single_scale(Generators[:scale_num],z_opts[:scale_num],NoiseAmp[:scale_num],G_curr,
                                                 reals[:scale_num+1],vgg,scale_out_dir, scale_factor, opt, centers=centers)

        torch.save(G_curr.state_dict(), os.path.join(scale_out_dir, G_WEIGHTS_FILE_NAME))
        [p.requires_grad_(False) for p in G_curr.parameters()]
        G_curr.eval()

        Generators[scale_num] = G_curr
        z_opts[scale_num] = z_opt
        NoiseAmp[scale_num] = noise_amp

        output_handler.save_network(Generators,z_opts,NoiseAmp,reals,out_dir)

        del G_curr
    return Generators, z_opts, NoiseAmp, reals
