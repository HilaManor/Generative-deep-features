"""A test script for generating random samples at different sizes and aspects
"""

from config import get_arguments
import torch
import random
import os
import image_helpers
import image_processing
import output_handler
import plotting_helpers
import tests
import training
import re
import json

def get_unique_name(path, name):
    possible_name = os.path.join(path, name)

    files = [os.path.basename(f) for f in os.listdir(path) if not os.path.isdir(os.path.join(path, f))]
    matches = re.findall('(' + name + r'(\((\d+)\))?)', '\n'.join(files))
    if len(matches):
        int_matches = [int(j) for x,i,j in matches if j]
        if int_matches:
            name += f'({max(int_matches)+1})'
        else:
            name += '(1)'

    return name

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--trained_net_dir', help='trained network folder', required=True)
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)
    parser.add_argument('--amount', type=int, default=1, help='the amount of samples to be generated in the '
                                                              'given resize factors.')
    parser.add_argument('--verbose', action='store_true', help='Output the generated samples across all the '
                                                              'passages in the different scales')
    opt = parser.parse_args()
    
    # Load the trained model parameters according to the params.txt file in the folder
    opt = output_handler.load_parameters(opt, opt.trained_net_dir)

    basename = os.path.basename(opt.image_path)
    basename = basename[:basename.rfind('.')]

    real_img = image_helpers.read_image(opt.image_path, opt.nc, opt.is_cuda)
    real_resized, scale_factor, total_scales = image_processing.preprocess_image(real_img, opt)
    opt.nzx = real_resized.shape[2]
    opt.nzy = real_resized.shape[0]

    out_dir = os.path.join(opt.trained_net_dir, 'Scaled_samples')
    if opt.verbose:
        out_dir = os.path.join(out_dir, str(opt.manual_seed))
    os.makedirs(out_dir, exist_ok=True)
    
    Generators, z_opts, NoiseAmp, reals = output_handler.load_network(opt.trained_net_dir)
   
    for i in range(opt.amount):
        out = tests.generate_random_sample(Generators, z_opts, scale_factor, NoiseAmp,
                                       reals, opt=opt, scale_h=opt.scale_h, scale_v=opt.scale_v)
        if opt.verbose:
            name = get_unique_name(out_dir, f'scaled_im_v{opt.scale_v}_h{opt.scale_h}')
            for idx, out_im in enumerate(out):
                plotting_helpers.save_im(out_im, out_dir, f'{name}_S{idx}', convert=True)
        else:
            plotting_helpers.save_im(out[-1], out_dir, get_unique_name(out_dir, f'scaled_im_v{opt.scale_v}'
                                                                                f'_h{opt.scale_h}_'
                                                                                f'{opt.manual_seed}'), 
                                     convert=True)
