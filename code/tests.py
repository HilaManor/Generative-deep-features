"""Test functions to run over the trained network.
function run_tests - run all the available tests
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import torch
from torch import nn
import wandb
import models

import image_processing
from plotting_helpers import convert_im, save_im


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_tests(generators, z_opts, scale_factor, noise_amps, real_imgs, out_dir, opt):
    """Run all the available tests on the trained generators.
    Results are outputted at '<out_dir>/tests'

    :param generators: list of trained generators
    :param z_opts: list of padded optimal reconstruction noise(z)
    :param scale_factor: the actual (calculated) scaling factor between scales
    :param noise_amps: list of noise multipliers (amplitudes) with which each scale's generator was
                       trained
    :param real_imgs: list of the real image downscaled at each scale
    :param out_dir: the output base folder to create the test dir in
    :param opt: the configuration parameters for the network
    :return: None
    """
    tests_path = os.path.join(out_dir, 'tests')
    os.makedirs(tests_path, exist_ok=True)

    wandb_res = {}
    # Test 1 - Propagate an image through the different scales
    for sample_i in range(opt.generate_fake_amount):
        results = generate_random_sample(generators, z_opts, scale_factor, noise_amps, real_imgs, opt)
        wandb_res[f'Generated Sample {sample_i}'] = []
        for scale, res in enumerate(results):
            save_im(res, tests_path, f'Sample{sample_i}_S{scale}', convert=True)
            im = wandb.Image(convert_im(res), caption=f'Sample{sample_i}_S{scale}')
            wandb_res[f'Generated Sample {sample_i}'].append(im)

    # Test 2 - Propegate an image only from some scale upwards
    for gen_start_scale in [1]:
        wandb_res[f'Generated from Scale {gen_start_scale}'] = []
        for sample_i in range(3):
            results = generate_random_sample(generators, z_opts, scale_factor, noise_amps,
                                             real_imgs, opt, gen_start_scale=gen_start_scale)
            res = results[-1]
            save_im(res, tests_path, f'Sample_startGen{gen_start_scale}_{sample_i}', convert=True)
            im = wandb.Image(convert_im(res),
                             caption=f'Sample_startGen{gen_start_scale}_{sample_i}')
            wandb_res[f'Generated from Scale {gen_start_scale}'].append(im)

    # Test 3 - Propegate an image only from some scale upwards
    scaled_path = os.path.join(tests_path, 'scaled')
    os.makedirs(scaled_path, exist_ok=True)
    wandb_res[f'Aspect V2H1'] = []
    wandb_res[f'Aspect H2V1'] = []
    for sample_i in range(opt.generate_fake_amount):
        horz_results = generate_random_sample(generators, z_opts, scale_factor, noise_amps,
                                              real_imgs, opt, scale_h=2, scale_v=1)
        vert_results = generate_random_sample(generators, z_opts, scale_factor, noise_amps,
                                              real_imgs, opt, scale_h=1, scale_v=2)
        horz_res = horz_results[-1]
        vert_res = vert_results[-1]
        save_im(horz_res, scaled_path, f'H2_V1_Sample{sample_i}', convert=True)
        save_im(vert_res, scaled_path, f'V2_H1_Sample{sample_i}', convert=True)

        im_h = wandb.Image(convert_im(horz_res), caption=f'H2_V1_Sample{sample_i}')
        im_v = wandb.Image(convert_im(vert_res), caption=f'V2_H1_Sample{sample_i}')
        wandb_res[f'Aspect H2V1'].append(im_h)
        wandb_res[f'Aspect V2H1'].append(im_v)

    wandb.log(wandb_res)

def generate_random_sample(generators, z_opts, scale_factor, noise_amps, real_imgs, opt,
                           gen_start_scale=0, n=0, scale_v=1, scale_h=1, fake=None):
    """Generate a single sample, and track it through the scales

    :param generators: list of trained generators
    :param z_opts: list of padded optimal reconstruction noise(z)
    :param scale_factor: the actual (calculated) scaling factor between scales
    :param noise_amps: list of noise multipliers (amplitudes) with which each scale's generator was
                       trained
    :param real_imgs: list of the real image downscaled at each scale
    :param opt: the configuration parameters for the network
    :param gen_start_scale: the scale to start the "fresh" generation from
    :return: list of fake images of the same sample at different scales
    """

    pad_amount = models.get_pad_amount(opt.ker_size, opt.num_layer, opt.pad_type)
    pad_func = nn.ZeroPad2d(int(pad_amount))

    results = []

    if fake is None:
        # the initial prev is a zero image (add nothing to the noise...)
        fake = torch.full((real_imgs[0].shape[0], 
                           real_imgs[0].shape[1], 
                           round(real_imgs[0].shape[2] * scale_v), 
                           round(real_imgs[0].shape[3] * scale_h)), 
                          0, device=opt.device)

    for i, (G, z_opt, real_img, noise_amp) in enumerate(zip(generators, z_opts, real_imgs,
                                                            noise_amps)):
        nzy = round((z_opt.shape[2] - pad_amount * 2) * scale_v)
        nzx = round((z_opt.shape[3] - pad_amount * 2) * scale_h)

        # Only in the first scale the noise should be equal in all the color channels
        if n:
            z = image_processing.generate_noise([opt.nc, nzy, nzx], device=opt.device)
        else:
            z = image_processing.generate_noise([1, nzy, nzx], device=opt.device)
            z = z.expand(1, 3, z.shape[2], z.shape[3])
        z = pad_func(z)

        if n < gen_start_scale:
            z = z_opt

        # the last fake image has been upscaled, and so we cut it to fit perfectly the cur image
        prev_fake = fake[:, :, :round(scale_v * real_img.shape[2]), :round(scale_h * real_img.shape[3])]
        prev_fake = pad_func(prev_fake)
        z = z[:,:,:prev_fake.shape[2], :prev_fake.shape[3]]

        z_in = noise_amp * z + prev_fake
        fake = G(z_in.detach(), prev_fake)
        results.append(fake)

        # prepare for next scale
        fake = image_processing.resize(fake, 1 / scale_factor, opt.nc, opt.is_cuda)
        n += 1

    return results
