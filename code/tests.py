"""Test functions to run over the trained network.
function run_tests - run all the available tests
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import torch
from torch import nn
import wandb
import image_processing
from plotting_helpers import convert_im, save_im


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_tests(generators, Zs, scale_factor, noise_amps, real_imgs, out_dir, opt):
    """Run all the available tests on the trained generators.
    Results are outputted at '<out_dir>/tests'

    :param generators: list of trained generators
    :param Zs: list of padded optimal reconstruction noise(z)
    :param scale_factor: the actual (calculated) scaling factor between scales
    :param noise_amps: list of noise multipliers (amplitudes) with which each scale's generator was
                       trained
    :param real_imgs: list of the real image downscaled at each scale
    :param out_dir: the output base  folder tto  create the test dir in
    :param opt: the configuration parameters for the network
    :return: None
    """
    tests_path = os.path.join(out_dir, 'tests')
    os.makedirs(tests_path, exist_ok=True)

    # Test 1 - Propagate an image through the different scales
    wandb_res = {}
    for sample_i in range(5):
        results = _generate_random_sample(generators, Zs, scale_factor, noise_amps, real_imgs, opt)
        wandb_res[f'Generated Sample {sample_i}'] = []
        for scale, res in enumerate(results):
            save_im(res, tests_path, f'Sample{sample_i}_S{scale}', convert=True)
            im = wandb.Image(convert_im(res), caption=f'Sample{sample_i}_S{scale}')
            wandb_res[f'Generated Sample {sample_i}'].append(im)
    wandb.log(wandb_res)


def _generate_random_sample(generators, Zs, scale_factor, noise_amps, real_imgs, opt):
    """Generate a single sample, and track it through the scales

    :param generators: list of trained generators
    :param Zs: list of padded optimal reconstruction noise(z)
    :param scale_factor: the actual (calculated) scaling factor between scales
    :param noise_amps: list of noise multipliers (amplitudes) with which each scale's generator was
                       trained
    :param real_imgs: list of the real image downscaled at each scale
    :param opt: the configuration parameters for the network
    :return: list of fake images of the same sample at different scales
    """

    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_func = nn.ZeroPad2d(int(pad_noise))

    results = []
    # the initial prev is a zero image (add nothing to the noise...)
    fake = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
    for i, (G, Z_opt, real_img, noise_amp) in enumerate(zip(generators, Zs, real_imgs,
                                                            noise_amps)):
        nzx = Z_opt.shape[2] - pad_noise * 2
        nzy = Z_opt.shape[3] - pad_noise * 2

        # Only in the first scale the noise should be equal in all the color channels
        if i:
            z = image_processing.generate_noise([opt.nc, nzx, nzy], device=opt.device)
        else:
            z = image_processing.generate_noise([1, nzx, nzy], device=opt.device)
            z = z.expand(1, 3, z.shape[2], z.shape[3])
        z = pad_func(z)

        # the last fake image has been upscaled, and so we cut it to fit perfectly the cur image
        prev_fake = fake[:, :, :real_img.shape[2], :real_img.shape[3]]
        prev_fake = pad_func(prev_fake)

        z_in = noise_amp * z + prev_fake
        fake = G(z_in.detach(), prev_fake)
        results.append(fake)

        # prepare for next scale
        fake = image_processing.resize(fake, 1 / scale_factor, opt.nc, opt.is_cuda)

    return results
