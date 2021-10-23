import os
import re
import torch
import json
import random


def __generate_out_name(opt):
    """
	The function generates base name for files based on the current opt.
	
	:param opt: The current configure of the test.
	
	:return: string of the base name, with importent data from the opt.
	"""
    base_name = f"{opt.loss_func}_e{opt.epochs}_{opt.nzy}px_lr{opt.lr}" \
                 f"a{opt.alpha}_up-{opt.chosen_layers[-1]}_W{opt.layers_weights}"
    return base_name


def gen_unique_out_dir_path(base_out, baseim_name, opt):
    """
	The function generates unique name for a new file, to avoid override.
	
	:param base_out: The base directory path.
	:param baseim_name: The base image name.
	:param opt: The current configure of the test.
	
	:return: string of the possible path, with importent data from the opt.
	"""
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
    """
	The function generates a folder for the current scale. If the folder already
		exists, only a path is returned.
	
	:param out_dir: The current output dir of the test.
	:param scale: The current scale.
	
	:return: The function returns a string for the output directory.
	"""
    path = os.path.join(out_dir, str(scale))
    os.makedirs(path, exist_ok=True)
    return path

def save_network(Generators, z_opts, noise_amps, real_imgs, out_dir):
    """
	The function saves the current generators,noises and real images. 
	
	:param Generators: Trained generators to save.
	:param z_opts: The noise patters for reconstructing the real images.
	:param noise_amps: Noise amp that was calculated during training process.
	:param real_imgs: Re-scaled real images array.
	:param out_dir: The current output dir of the test for saving the data.
	
	"""
    torch.save(Generators, os.path.join(out_dir, 'Generators.pth'))
    torch.save(z_opts, os.path.join(out_dir, 'z_opts.pth'))
    torch.save(noise_amps, os.path.join(out_dir, 'noise_amps.pth'))
    torch.save(real_imgs, os.path.join(out_dir, 'real_imgs.pth'))


def load_network(input_dir):
    """
	The function loads generators,noises and real images from the given directory.
	
	:param input_dir: The directory to read the files from. must contain the 
			files: 'Generators.pth', 'z_opts.pth', 'noise_amps.pth',
			'real_imgs.pth'  for loading all the data.

	:return: The function returns the loaded objects: Generators, z_opts,
			noise_amps, real_imgs
	"""

    Generators = torch.load(os.path.join(input_dir, 'Generators.pth'))
    z_opts = torch.load(os.path.join(input_dir, 'z_opts.pth'))
    noise_amps = torch.load(os.path.join(input_dir, 'noise_amps.pth'))
    real_imgs = torch.load(os.path.join(input_dir, 'real_imgs.pth'))
    return Generators, z_opts, noise_amps, real_imgs

def load_parameters(opt, dir):
    """
	The function loads a new configuration file
	
	:param opt: The current configure of the test.
	:param dir: The directory to read 'params.txt', where the old opt file was 
			saved.
			
	:return: The function updated configuration object opt.
	"""
    with open(os.path.join(dir, 'params.txt'), 'r') as f:
        old_opt_dict = json.load(f)
        mseed = opt.manual_seed
        opt.__dict__.update(old_opt_dict)
        if mseed is None:
            opt.manual_seed = random.randint(1, 10000)
        else:
            opt.manual_seed = mseed
        opt.device = torch.device(opt.device)
        print("Random Seed: ", opt.manual_seed)
        random.seed(opt.manual_seed)
        torch.manual_seed(opt.manual_seed)
    return opt