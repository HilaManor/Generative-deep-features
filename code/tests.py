from torch import nn
import torch
import image_processing
import os
import wandb
import plotting_helpers

def run_tests(Generators, Zs, scale_factor, noise_amps, real_imgs, out_dir, opt):
    tests_path = os.path.join(out_dir, 'tests')
    results = generate_random_samples(Generators, Zs, scale_factor, noise_amps, real_imgs, opt, n=5)

    os.makedirs(tests_path, exist_ok=True)
    wandb_res = {}
    for sample_i, sample in enumerate(results):
        wandb_res[f'Generated Sample {sample_i}'] = []
        for scale, res in enumerate(sample):
            plotting_helpers.save_im(res, tests_path, f'Sample{sample_i}_S{scale}', convert=True)
            im = wandb.Image(plotting_helpers.convert_im(res), caption=f'Sample{sample_i}_S{scale}')
            wandb_res[f'Generated Sample {sample_i}'].append(im)
    wandb.log(wandb_res)

def generate_random_samples(Generators, Zs, scale_factor, noise_amps, real_imgs, opt, n=5):
    """

    :param Generators:
    :param Zs:
    :param scale_factor:
    :param noise_amps:
    :param real_imgs:
    :param opt:
    :param n:
    :return:
    """
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_func = nn.ZeroPad2d(int(pad_noise))

    results = []
    for sample in range(n):
        ims = []
        fake = torch.full([1, opt.nc, opt.nzx, opt.nzy], 0, device=opt.device)
        for i, (G, Z_opt, real_img, noise_amp) in enumerate(zip(Generators, Zs, real_imgs, noise_amps)):
            nzx = Z_opt.shape[2] - pad_noise*2
            nzy = Z_opt.shape[3] - pad_noise*2

            if i:
                z = image_processing.generate_noise( [opt.nc, nzx, nzy], device=opt.device)
            else:
                z = image_processing.generate_noise([1, nzx, nzy], device=opt.device)
                z = z.expand(1, 3, z.shape[2], z.shape[3])
            z = pad_func(z)

            prev_fake = fake[:, :, :real_img.shape[2], :real_img.shape[3]]
            prev_fake = pad_func(prev_fake)

            z_in = noise_amp * z + prev_fake
            fake = G(z_in.detach(), prev_fake)
            ims.append(fake)
            if i < len(Generators) - 1:
                fake = image_processing.resize(fake, 1 / scale_factor, opt.nc, opt.is_cuda)
                fake = fake[:, :, :real_imgs[i+1].shape[2], :real_imgs[i+1].shape[3]]
        results.append(ims)
    return results