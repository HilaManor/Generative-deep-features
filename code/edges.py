"""A test script for testing the variability of the generatted images
"""

from config import get_arguments
import image_processing
import image_helpers
import plotting_helpers
import numpy as np
import tests
import os
import re
import output_handler
import matplotlib.pyplot as plt
from skimage import color

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--trained_net_dir', help='trained network folder', required=True)
    parser.add_argument('--amount', type=int, default=100, help='the amount of images to '
                                                                'create and calculate the '
                                                                'standard variation on')
    opt = parser.parse_args()
    
    # Load the trained model parameters according to the params.txt file in the folder
    opt = output_handler.load_parameters(opt, opt.trained_net_dir)

    basename = os.path.basename(opt.image_path)
    basename = basename[:basename.rfind('.')]

    real_img = image_helpers.read_image(opt.image_path, opt.nc, opt.is_cuda)
    real_resized, scale_factor, total_scales = image_processing.preprocess_image(real_img, opt)
    opt.nzx = real_resized.shape[2]
    opt.nzy = real_resized.shape[0]

    out_dir = os.path.join(opt.trained_net_dir, 'Edges')
    os.makedirs(out_dir, exist_ok=True)

    Generators, z_opts, NoiseAmp, reals = output_handler.load_network(opt.trained_net_dir)
    
    ims = []
    for i in range(opt.amount):
        out = tests.generate_random_sample(Generators, z_opts, scale_factor, NoiseAmp,
                                       reals, opt=opt)
        #ims.append(color.rgb2gray(plotting_helpers.convert_im(out[-1])))
        plotting_helpers.save_im(out[-1], out_dir, f"seed_{opt.manual_seed}_im_{i}",
                                 convert=True)
        ims.append((256*color.rgb2gray(plotting_helpers.convert_im(out[-1]))).astype('uint8'))
        print(f'{i}/{opt.amount}', end='\r')

    ims = np.stack(ims, axis=-1)
    #ims_var = np.var(ims, axis=3)
    ims_var = np.std(ims, axis=2)
    im = plt.imshow(ims_var, cmap='jet', vmin=0)
    #im = plt.imshow(ims_var, cmap='jet', vmin=0, vmax=0.3)
    plt.colorbar(im)
    plt.savefig(os.path.join(opt.trained_net_dir, f'edges_{opt.amount}.png'))