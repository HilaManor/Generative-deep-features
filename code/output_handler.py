import os
import re
import torch

def __generate_out_name(opt):
    base_name = f"{opt.loss_func}_e{opt.epochs}_{opt.nzx}px_lr{opt.lr}" \
                 f"a{opt.alpha}_up-{opt.chosen_layers[-1]}_W{opt.layers_weights}"
    return base_name


def gen_unique_out_dir_path(base_out, baseim_name, opt):
    possible_name = __generate_out_name(opt)
    possible_base = os.path.join(base_out, baseim_name)
    possible_path = os.path.join(possible_base, possible_name)

    if os.path.exists(possible_path):
        # rename with "name_name(num)"
        dirs = [f for f in os.listdir(possible_base) if os.path.isdir(os.path.join(possible_base, f))]

        ptrn = possible_name.replace('[', '\[').replace(']', '\]')
        matches = re.findall(ptrn+r'(\((\d+)\))?', '\n'.join(dirs))
        int_matches = [int(j) for i,j in matches if j]
        if int_matches:
            possible_name += f'({max(int_matches)+1})'
        else:
            possible_name += '(1)'

        possible_path = os.path.join(possible_base, possible_name)
    return possible_path

def __generate_out_name(opt):
    base_name = f"{opt.loss_func}_e{opt.epochs}_{opt.nzy}px_lr{opt.lr}" \
                 f"a{opt.alpha}_up-{opt.chosen_layers[-1]}_W{opt.layers_weights}"
    return base_name


def gen_unique_out_dir_path(base_out, baseim_name, opt):
    possible_name = __generate_out_name(opt)
    possible_base = os.path.join(base_out, baseim_name)
    possible_path = os.path.join(possible_base, possible_name)

    if os.path.exists(possible_path):
        # rename with "name_name(num)"
        dirs = [f for f in os.listdir(possible_base) if os.path.isdir(os.path.join(possible_base, f))]

        ptrn = possible_name.replace('[', '\[').replace(']', '\]')
        matches = re.findall(ptrn+r'(\((\d+)\))?', '\n'.join(dirs))
        int_matches = [int(j) for i,j in matches if j]
        if int_matches:
            possible_name += f'({max(int_matches)+1})'
        else:
            possible_name += '(1)'

        possible_path = os.path.join(possible_base, possible_name)
    return possible_path


def gen_scale_dir(out_dir, scale):
    path = os.path.join(out_dir, str(scale))
    os.makedirs(path, exist_ok=True)
    return path

def save_network(Generators, z_opts, noise_amps, real_imgs, out_dir):
    torch.save(Generators, os.path.join(out_dir, 'Generators.pth'))
    torch.save(z_opts, os.path.join(out_dir, 'z_opts.pth'))
    torch.save(noise_amps, os.path.join(out_dir, 'noise_amps.pth'))
    torch.save(real_imgs, os.path.join(out_dir, 'real_imgs.pth'))


def load_network(input_dir):
    Generators = torch.load(os.path.join(input_dir, 'Generators.pth'))
    z_opts = torch.load(os.path.join(input_dir, 'z_opts.pth'))
    noise_amps = torch.load(os.path.join(input_dir, 'noise_amps.pth'))
    real_imgs = torch.load(os.path.join(input_dir, 'real_imgs.pth'))
    return Generators, z_opts, noise_amps, real_imgs

